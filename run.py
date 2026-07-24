#!/usr/bin/env python3
"""Run the voice assistant with microphone input and speaker output.

Prerequisites
-------------
1. Set environment variables::

    export DASHSCOPE_API_KEY="your_aliyun_dashscope_key"

   For Baidu TTS (legacy), also set::

    export BAIDU_TTS_API_KEY="your_baidu_api_key"
    export BAIDU_TTS_SECRET_KEY="your_baidu_secret_key"

2. Start the nanobot gateway (default ``ws://127.0.0.1:8765/``).

3. Run::

    python run.py

Configuration
-------------
Edit the parameters below to match your setup.
"""

import asyncio
import os
import sys

# Make the package importable when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant import VoiceAssistant
from voice_assistant.adapter import AudioDeviceAdapter
from voice_assistant.kws import KeywordWakeDetector


# ================================================================
# Configuration — adjust these to your environment
# ================================================================

# STT: Aliyun DashScope
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
STT_LANGUAGE = "zh"

# TTS: Alibaba Cloud CosyVoice (DashScope)
# Model: cosyvoice-v3-flash = fastest, cosyvoice-v3-plus = highest quality
TTS_PROVIDER = "baidu"  # "cosyvoice" or "baidu"
TTS_MODEL = "cosyvoice-v3-flash"
TTS_VOICE = "longanhuan"  # 龙安欢 (cheerful female, works with cosyvoice-v3-flash)
# Other voice options: longanchen (male), longanhuan (cheerful), longanxia (warm)
# Baidu TTS fallback (only used if TTS_PROVIDER="baidu")
BAIDU_TTS_API_KEY = os.getenv("BAIDU_TTS_API_KEY", "")
BAIDU_TTS_SECRET_KEY = os.getenv("BAIDU_TTS_SECRET_KEY", "")
TTS_SPEAKER = "1"  # Baidu speaker id

# LLM: nanobot → DeepSeek
NANOBOT_WS_URL = "ws://127.0.0.1:8765/"
NANOBOT_TOKEN = "litests-shared-secret"
LLM_SYSTEM_PROMPT = "你叫小牛，是一个友好的语音助手，用简洁干练的中文回答用户的问题。除非用户明确说**帮我查一下**，否者不调用工具或思考模式，直接回答"

# VAD
VAD_VOLUME_THRESHOLD = -20  # dB, lower = more sensitive
VAD_SILENCE_THRESHOLD = 0.5  # seconds

# Wake / interaction
WAKEWORDS = ["小牛"]  # Set to None to always stay awake
END_WORDS = [
    # 告别
    "再见", "拜拜", "拜", "回头见", "下次见", "一会见", "晚点聊",
    "明天见", "晚安", "回见", "走啦", "先走了",
    # 命令停止
    "结束", "停止", "停", "停下", "别说了", "住口", "住嘴",
    "闭嘴", "别吵了", "别吵", "别闹了", "别念叨了", "别逼逼了",
    # 退下 / 赶人
    "退下", "退下吧", "下去吧", "下去", "没你事了", "你可以走了",
    "你走吧", "你退下", "你可以退下了", "跪安吧", "散了吧",
    "退开", "一边去", "闪一边去",
    # 去休息 / 去玩
    "去玩吧", "去休息吧", "歇着吧", "睡觉去吧", "睡吧",
    "休息吧", "歇着", "去歇着", "去睡觉", "你睡吧",
    # 不需要了
    "不用了", "没事了", "就这样", "就这样吧", "行了好吧",
    "行了行了", "好了好了", "好嘞", "好了", "行了吧",
    "知道了", "了解", "懂了", "收到", "已阅",
    # 结束标记
    "先这样", "先到这", "就到这", "到这吧", "差不多了",
    "够了", "可以了", "行了", "打住", "得了吧",
    # 闲聊式结束
    "忙你的去吧", "你忙吧", "不打扰你了", "你继续",
    "跪安", "朕知道了", "爱卿退下",
]
AWAKE_TIMEOUT = 600.0  # seconds
CONTEXT_TIMEOUT = 3600.0  # 1 hour

# Audio device
ECHO_CANCELLATION = True  # Mute mic while playing AI speech

# KWS (local keyword spotting with sherpa-onnx)
# Set to None to disable and fall back to cloud-STT-based wake word detection
KWS_ENABLED = True
KWS_KEYWORDS = ["小牛"]       # Keywords to detect locally
KWS_MODEL_DIR = None          # None = use the default bundled model
KWS_THRESHOLD = 0.25          # Detection confidence threshold (lower = more sensitive)


# ================================================================
# Main
# ================================================================

async def main():
    # --- KWS (local keyword spotting) ---
    kws = None
    if KWS_ENABLED:
        print(f"🔑 Loading local KWS model for keywords: {KWS_KEYWORDS}")
        kws = KeywordWakeDetector(
            keywords=KWS_KEYWORDS,
            model_dir=KWS_MODEL_DIR,
            keywords_threshold=KWS_THRESHOLD,
            debug=True,
        )
        print(f"✅ KWS ready (offline wake-word detection)")
    else:
        print("⚠️  KWS disabled — using cloud STT for wake-word detection (not cost-effective)")

    # --- Build pipeline ---
    assistant = VoiceAssistant(
        kws=kws,
        stt_api_key=DASHSCOPE_API_KEY,
        stt_language=STT_LANGUAGE,
        tts_api_key=BAIDU_TTS_API_KEY if TTS_PROVIDER == "baidu" else DASHSCOPE_API_KEY,
        tts_provider=TTS_PROVIDER,
        tts_model=TTS_MODEL,
        tts_voice=TTS_VOICE,
        tts_secret_key=BAIDU_TTS_SECRET_KEY,  # only used if provider="baidu"
        tts_speaker=TTS_SPEAKER,
        nanobot_ws_url=NANOBOT_WS_URL,
        nanobot_token=NANOBOT_TOKEN,
        llm_system_prompt=LLM_SYSTEM_PROMPT,
        vad_volume_threshold=VAD_VOLUME_THRESHOLD,
        vad_silence_threshold=VAD_SILENCE_THRESHOLD,
        wakewords=WAKEWORDS,
        end_words=END_WORDS,
        awake_timeout=AWAKE_TIMEOUT,
        context_timeout=CONTEXT_TIMEOUT,
        debug=True,
    )

    # --- Wire up audio I/O ---
    adapter = AudioDeviceAdapter(assistant, cancel_echo=ECHO_CANCELLATION)

    # --- Start listening (blocks until Ctrl+C) ---
    print("\n🎤 Voice assistant ready. Press Ctrl+C to stop.\n")
    try:
        await adapter.start_listening("default_user")
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        await assistant.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

