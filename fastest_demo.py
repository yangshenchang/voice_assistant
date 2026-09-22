#!/usr/bin/env python3
"""fastest_demo.py — 极简 STT + LLM + TTS 全链路延迟测试 demo

链路: 麦克风 → VAD切句 → STT(qwen3-asr-flash) → LLM(deepseek-flash 流式, 显式禁用思考)
      → TTS(cosyvoice-v3-flash 流式) → 扬声器

打印两个关键时延:
  [TTFT] LLM 流式返回首个字符(对应工程 pipeline.py 中 stt_to_llm_first_ms 的语义),
         并额外给出「说话结束→首字」的总开销
  [TTS ] 首个音频包到达的时刻 —— 流式播放, 包一到就出声, 也就是用户真正听到第一个字的时刻

依赖: dashscope, httpx, PyAudio。 用法: python fastest_demo.py
      → 对着麦克风说话(停顿 0.5s 判为一句话结束) → 看打印耗时, Ctrl+C 退出。
"""

import asyncio
import json
import os
import re
import sys
import time

import dashscope
import httpx
import pyaudio
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

# 复用工程内的现成模块(VAD切句 / 阿里云ASR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stt import AliyunASRSpeechRecognizer
from vad import StandardSpeechDetector

# ============================================================
# 配置
# ============================================================
# LLM: DeepSeek 官方 key — 用于 deepseek-flash
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# STT/TTS: 阿里云百炼 key — 和 .vscode/launch.json 里的 DASHSCOPE_API_KEY 同一个
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 密钥只从环境变量读取, 不写进代码库(仓库是公开的)
if not LLM_API_KEY:
    raise SystemExit("缺少环境变量 DEEPSEEK_API_KEY, 请先 export 后再运行")
if not DASHSCOPE_API_KEY:
    raise SystemExit("缺少环境变量 DASHSCOPE_API_KEY, 请先 export 后再运行")

# LLM: DeepSeek 官方 OpenAI 兼容端点, 直接调 deepseek-flash(不经过 nanobot)
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-flash"

SYSTEM_PROMPT = "你是小牛, 一个语音助手。用最简短的中文口语直接回答, 禁止 Markdown 符号和 emoji。"

# STT / VAD
STT_LANG = "zh"
VAD_VOLUME_DB = -20        # 越灵敏值越小; 静音判定灵敏度
VAD_SILENCE = 0.5          # 静音秒数, 够了就判定一句话结束

# TTS: CosyVoice v3-flash(百炼, 和 STT 同一个 key)
TTS_MODEL = "cosyvoice-v3-flash"
TTS_VOICE = "longanhuan"   # 龙安欢(活泼女声)
TTS_SAMPLE_RATE = 16000


# ============================================================
# LLM: 最简流式客户端 (OpenAI 兼容 /chat/completions + SSE)
# ============================================================
class StreamingLLM:
    def __init__(self, base_url, api_key, model):
        self.url = f"{base_url}/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        # client 只建一次并复用连接。若像以前那样每次请求都 httpx.AsyncClient():
        # 它内部的 ssl 建上下文是同步执行的, 本机要 ~3s, 整轮对话凭空多等 3s。
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, text):
        """逐 token 产出 LLM 文本。调用方在收到第一个 token 时记录首字响应时间。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": True,
            "thinking": {"type": "disabled"},  # 显式禁止思考模式: 跳过思维链, 首字立即返回
            "temperature": 0.3,
            "max_tokens": 300,
        }
        async with self.client.stream(
            "POST", self.url, json=payload, headers=self.headers
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta


# ============================================================
# TTS: cosyvoice-v3-flash 流式合成 + 流式播放
# ============================================================
class _Speaker(ResultCallback):
    """TTS 回调: 每个音频包一到就写进扬声器, 并记下首个包的到达时刻。"""

    def __init__(self, stream, first: list):
        self._stream = stream
        self._first = first

    def on_data(self, data: bytes):
        if not self._first:
            self._first.append(time.perf_counter())
        self._stream.write(data)


class CosyVoiceTTS:
    """一句一次合成: 首个音频包约 0.6~0.8s 到达, 到达即出声(不等整句合成完)。"""

    def __init__(self, api_key, model, voice):
        dashscope.api_key = api_key
        self.model = model
        self.voice = voice
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16, channels=1, rate=TTS_SAMPLE_RATE, output=True,
        )

    def speak(self, text):
        """阻塞合成并播放一句(返回时该句音频已全部写入扬声器)。

        返回首个音频包到达的时刻(perf_counter)。
        注: 这里用 streaming_call + streaming_complete, 不用 SDK 的 call()
            —— 本机 dashscope 1.25.14 上 call() 只会停在 task-started, 拿不到音频。
        """
        first = []
        synth = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            callback=_Speaker(self.stream, first),
        )
        synth.streaming_call(text)
        synth.streaming_complete()
        return first[0] if first else None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


# ============================================================
# 工具
# ============================================================
def split_sentences(text):
    """按中文句末标点切句(与工程 LLMService.split_chars 一致)。"""
    return [s for s in re.split(r"(?<=[。！？.!?])", text.strip()) if s.strip()]


# ============================================================
# 单轮: STT → LLM → TTS (含首字响应时间 / 首音频时刻打印)
# ============================================================
async def process_turn(audio_bytes: bytes, seg_dur: float, sid: str):
    t_end = time.perf_counter()  # T0: 说话结束(VAD切句)

    # ---- 1. STT
    text = await stt.transcribe(audio_bytes)
    t_stt = time.perf_counter()  # T1: STT 完成
    if not text:
        print(f"  [STT] (未识别到内容, 段长 {seg_dur:.2f}s)")
        return
    print(f"  [STT] \"{text}\"    {int((t_stt - t_end) * 1000)} ms")

    # ---- 2. LLM 流式 (记录首字时刻)
    full = ""
    t_first = None  # T2: 首个字符到达
    async for delta in llm.generate(text):
        if t_first is None:
            t_first = time.perf_counter()
            print(f"  [LLM] ⚡首字 <{delta}>", flush=True)
        full += delta
    if not full:
        print("  [LLM] (空回复)")
        return

    # ---- 首字响应时间 (工程语义: STT完成→首字; 另给说话结束→首字总开销)
    print(f"  [TTFT] 首字响应时间: STT完成→首字 {(t_first - t_stt) * 1000:6.0f} ms"
          f"  |  说话结束→首字 {(t_first - t_end) * 1000:6.0f} ms")
    print(f"  [LLM] 全文: {full}  ({len(full)} 字)")

    # ---- 3. TTS: 逐句合成→流式播放, 首句的首个音频包就是用户听到的第一个字
    for i, sent in enumerate(split_sentences(full)):
        t_pkt = await asyncio.to_thread(tts.speak, sent)
        if i == 0 and t_pkt:
            print(f"  [TTS] 首音频: 说话结束→出声 {int((t_pkt - t_end) * 1000)} ms")

    print(f"  [TOTAL] 说话结束→音频全部送完 {int((time.perf_counter() - t_end) * 1000)} ms")
    print("-" * 72)


# ============================================================
# main: 麦克风 → VAD
# ============================================================
async def main():
    global stt, llm, tts
    stt = AliyunASRSpeechRecognizer(api_key=DASHSCOPE_API_KEY, language=STT_LANG, debug=False)
    llm = StreamingLLM(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)
    tts = CosyVoiceTTS(DASHSCOPE_API_KEY, TTS_MODEL, TTS_VOICE)

    vad = StandardSpeechDetector(
        volume_db_threshold=VAD_VOLUME_DB,
        silence_duration_threshold=VAD_SILENCE,
    )
    vad.on_speech_detected(process_turn)

    pa = pyaudio.PyAudio()
    mic = pa.open(
        rate=16000, channels=1, format=pyaudio.paInt16,
        input=True, frames_per_buffer=512,
    )
    print("🎤 就绪。对着麦克风说话(停顿0.5s判为一句话结束), Ctrl+C 退出。\n")
    try:
        while True:
            chunk = mic.read(512, exception_on_overflow=False)
            await vad.process_samples(chunk, "demo")
            await asyncio.sleep(0.0001)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop_stream()
        mic.close()
        pa.terminate()
        tts.close()
        await llm.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
