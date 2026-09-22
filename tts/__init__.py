"""Text-to-Speech engines: Baidu TTS and Alibaba Cloud CosyVoice."""

import asyncio
import io
import logging
import os
import subprocess
import threading
import urllib.parse
import wave
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Tuple

import dashscope
import httpx

logger = logging.getLogger(__name__)


def wav_to_pcm(wav: bytes) -> Tuple[int, bytes]:
    """WAV → (采样率, 16bit PCM)。"""
    with wave.open(io.BytesIO(wav), "rb") as wf:
        return wf.getframerate(), wf.readframes(wf.getnframes())


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SpeechSynthesizer(ABC):
    """Abstract base for text-to-speech engines."""

    #: 流式合成输出的采样率(16bit 单声道 PCM)
    sample_rate: int = 16000

    def __init__(self, *, sample_rate: int = 16000, timeout: float = 10.0,
                 debug: bool = False):
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.debug = debug

    @abstractmethod
    async def synthesize(self, text: str, language: str = None) -> bytes:
        """Synthesize text to a complete WAV. Returns empty bytes on failure."""

    async def synthesize_stream(self, text: str, language: str = None
                                ) -> AsyncGenerator[bytes, None]:
        """流式合成, 逐段产出 16bit 单声道 PCM(采样率 = ``self.sample_rate``)。

        默认实现退化成"整句合成完再一次产出"; 真正的流式引擎(如 CosyVoice)
        覆盖它, 做到**首个音频包一到就 yield**, 让扬声器立刻出声。
        """
        if not text or not text.strip():
            return
        wav = await self.synthesize(text, language)
        if not wav:
            return
        try:
            rate, pcm = wav_to_pcm(wav)
        except wave.Error as e:
            logger.error(f"TTS returned non-WAV bytes: {e}")
            return
        self.sample_rate = rate
        if pcm:
            yield pcm

    async def warmup(self):
        """启动预热钩子(建连接等)。默认不做任何事。"""

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# CosyVoice TTS (Alibaba Cloud DashScope) — Recommended
# ---------------------------------------------------------------------------

class CosyVoiceSpeechSynthesizer(SpeechSynthesizer):
    """Text-to-speech via Alibaba Cloud DashScope CosyVoice (tts_v2 流式接口).

    走的是 ``dashscope.audio.tts_v2`` 的 websocket 流式合成:
    ``synthesize_stream()`` 在**首个音频包到达时立刻 yield**, 不再等整句合成完,
    这是"首字出声"最快的路径(实测首包 ~0.6s)。

    Requires a DashScope API key (same key used for STT).

    Parameters
    ----------
    api_key:
        DashScope API key. Defaults to ``DASHSCOPE_API_KEY`` env var.
    model:
        CosyVoice model name. ``cosyvoice-v3-flash`` is the fastest;
        ``cosyvoice-v3-plus`` is the highest quality.
    voice:
        Voice id. ``longanhuan`` (龙安欢, cheerful female) is the default
        and works with cosyvoice-v3 models. ``longanlingxi`` (龙安灵犀)
        works with qwen-audio-3.0-tts-flash model. Available voices
        depend on your DashScope account/model.
    speech_rate:
        Speech speed, range [0.5, 2.0]. Default 1.0.
    pitch_rate:
        Pitch adjustment, range [0.5, 2.0]. Default 1.0.
    volume:
        Volume, range [0, 100]. Default 50.
    sample_rate:
        Audio sample rate in Hz. Default 16000.
    timeout:
        HTTP timeout in seconds.
    debug:
        Enable verbose logging.
    """

    # CosyVoice language hints
    LANG_MAP = {
        "zh": "zh", "zh-CN": "zh", "zh-TW": "zh",
        "en": "en", "en-US": "en", "en-GB": "en",
        "ja": "ja", "ja-JP": "ja",
        "ko": "ko", "ko-KR": "ko",
    }

    def __init__(
        self,
        *,
        api_key: str = None,
        model: str = "cosyvoice-v3-flash",
        voice: str = "longanhuan",
        speech_rate: float = 1.1,
        pitch_rate: float = 1.0,
        volume: int = 50,
        sample_rate: int = 16000,
        timeout: float = 10.0,
        debug: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, timeout=timeout, debug=debug)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key is required for CosyVoice TTS. "
                "Set DASHSCOPE_API_KEY env var or pass api_key parameter."
            )
        self.model = model
        self.voice = voice
        self.speech_rate = speech_rate
        self.pitch_rate = pitch_rate
        self.volume = volume
        dashscope.api_key = self.api_key

    # -- synthesize --------------------------------------------------------

    async def synthesize(self, text: str, language: str = None) -> bytes:
        """整句合成成 WAV(流式接口的兜底/离线用法)。"""
        pcm = bytearray()
        async for chunk in self.synthesize_stream(text, language):
            pcm.extend(chunk)
        if not pcm:
            return b""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(pcm))
        return buf.getvalue()

    async def synthesize_stream(self, text: str, language: str = None
                                ) -> AsyncGenerator[bytes, None]:
        """流式合成: 通过 dashscope tts_v2 的 websocket, 首个音频包一到就 yield。

        这就是"首字出声"的关键 —— 不再等整句合成完, 实测首包 ~0.6s 到达。
        """
        if not text or not text.strip():
            return

        from dashscope.audio.tts_v2 import (
            AudioFormat,
            ResultCallback,
            SpeechSynthesizer as _TtsV2,
        )

        if self.debug:
            logger.info(f"CosyVoice TTS (stream): '{text}'")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        DONE = object()

        class _Callback(ResultCallback):
            def on_data(self, data: bytes):
                loop.call_soon_threadsafe(queue.put_nowait, data)

            def on_error(self, message):
                loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(str(message)))

        def worker():
            try:
                synth = _TtsV2(
                    model=self.model,
                    voice=self.voice,
                    format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                    volume=self.volume,
                    speech_rate=self.speech_rate,
                    pitch_rate=self.pitch_rate,
                    callback=_Callback(),
                )
                # 注意: 不能用 tts_v2 的 call() —— 在当前 SDK 上它只会停在
                # task-started 拿不到音频, 必须 streaming_call + streaming_complete
                synth.streaming_call(text)
                synth.streaming_complete()
            except Exception as e:                       # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, DONE)

        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item is DONE:
                    break
                if isinstance(item, Exception):
                    logger.error(f"CosyVoice TTS error: {item}")
                    break
                yield item
        finally:
            await task                        # 线程退出, 别留悬挂的 websocket

    async def warmup(self):
        """预热一条 websocket(首次建连要 ~0.3-0.5s, 别让它落在第一轮对话里)。"""
        try:
            async for _ in self.synthesize_stream("你好"):
                break
        except Exception as e:                            # noqa: BLE001
            logger.warning(f"CosyVoice warmup failed: {e}")

    # -- helpers -----------------------------------------------------------

    def _map_language(self, language: str) -> List[str]:
        """Map a language code to CosyVoice language hints."""
        if not language:
            return ["zh"]
        mapped = self.LANG_MAP.get(language) or self.LANG_MAP.get(language.split("-")[0])
        if not mapped:
            logger.warning(f"Unsupported TTS language '{language}', fallback to 'zh'")
            return ["zh"]
        return [mapped]


# ---------------------------------------------------------------------------
# Edge TTS (Microsoft Edge online TTS, free)
# ---------------------------------------------------------------------------

class EdgeTTSSpeechSynthesizer(SpeechSynthesizer):
    """Text-to-speech via Microsoft Edge's online TTS service (edge-tts).

    Free to use (no API key required), but requires network access to
    Microsoft's speech endpoints. Output is converted from MP3 to 16 kHz
    mono WAV via ffmpeg so the playback adapter can consume it directly.

    Parameters
    ----------
    voice:
        Edge TTS voice id. Chinese options include ``zh-CN-XiaoxiaoNeural``
        (warm female, default), ``zh-CN-YunxiNeural`` (lively male),
        ``zh-CN-YunjianNeural`` (male), ``zh-CN-XiaoyiNeural`` (female).
        See ``edge-tts --list-voices`` for all options.
    rate:
        Speaking rate adjustment, e.g. ``"+10%"``, ``"-20%"``.
    pitch:
        Pitch adjustment, e.g. ``"+10Hz"``.
    volume:
        Volume adjustment, e.g. ``"+0%"``.
    sample_rate:
        Output WAV sample rate in Hz (default 16000).
    debug:
        Enable verbose logging.
    """

    def __init__(
        self,
        *,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        sample_rate: int = 16000,
        debug: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, timeout=30.0, debug=debug)
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise ImportError(
                "edge-tts is required for Edge TTS. Install with: pip install edge-tts"
            )
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

    # -- synthesize --------------------------------------------------------

    async def synthesize_stream(self, text: str, language: str = None
                                ) -> AsyncGenerator[bytes, None]:
        """流式: edge-tts 吐 mp3 分片 → ffmpeg 管道边转边出 PCM。

        不等整句 mp3 收完 —— 转出来的 PCM 立刻交给扬声器。
        """
        if not text or not text.strip():
            return

        import edge_tts

        rate = self.sample_rate
        # -f mp3: 明确告诉 ffmpeg 输入是裸 mp3, 免去 probe 等待
        # -flush_packets 1: 否则 ffmpeg 会攒满内部缓冲(约 1s 的 PCM)才写管道
        proc = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-f", "mp3", "-i", "pipe:0",
                "-f", "s16le", "-ac", "1", "-ar", str(rate),
                "-acodec", "pcm_s16le", "-flush_packets", "1", "pipe:1",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        DONE = object()

        def pump():
            """把 ffmpeg 的 PCM 输出搬到 asyncio 队列(阻塞读, 所以放线程里)。

            用 os.read 而不是 stdout.read(n): 后者要凑满 n 字节才返回,
            会把首片推迟 100ms 级。
            """
            try:
                fd = proc.stdout.fileno()
                while True:
                    data = os.read(fd, 4096)
                    if not data:
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, data)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, DONE)

        threading.Thread(target=pump, daemon=True).start()
        try:
            if self.debug:
                logger.info(f"EdgeTTS (stream): '{text}'")
            communicate = edge_tts.Communicate(
                text, voice=self.voice, rate=self.rate,
                pitch=self.pitch, volume=self.volume,
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # 写管道可能阻塞, 丢到线程里, 别卡事件循环
                    await asyncio.to_thread(proc.stdin.write, chunk["data"])
            proc.stdin.close()          # 输入结束 → ffmpeg 冲出剩余 PCM

            while True:
                item = await queue.get()
                if item is DONE:
                    break
                yield item
        finally:
            if proc.poll() is None:
                proc.kill()
            await asyncio.to_thread(proc.wait)

    async def synthesize(self, text: str, language: str = None) -> bytes:
        """整句合成成 WAV。"""
        pcm = bytearray()
        async for chunk in self.synthesize_stream(text, language):
            pcm.extend(chunk)
        if not pcm:
            return b""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(pcm))
        return buf.getvalue()

    # -- helpers -----------------------------------------------------------

    def _mp3_to_wav(self, mp3: bytes) -> bytes:
        """MP3 → 16k 单声道 WAV(ffmpeg 解码, wave 模块补合法头)。"""
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", "pipe:0",
                    "-f", "s16le", "-ac", "1", "-ar", str(self.sample_rate),
                    "-acodec", "pcm_s16le", "pipe:1",
                ],
                input=mp3,
                capture_output=True,
            )
        except FileNotFoundError:
            logger.error("ffmpeg not found — required to convert EdgeTTS MP3 to WAV")
            return b""

        if proc.returncode != 0 or not proc.stdout:
            logger.error(
                f"ffmpeg conversion failed: {proc.stderr.decode(errors='replace')[:200]}"
            )
            return b""

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(proc.stdout)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Baidu TTS (legacy)
# ---------------------------------------------------------------------------

class BaiduSpeechSynthesizer(SpeechSynthesizer):
    """Text-to-speech via Baidu TTS API.

    Requires Baidu API key / secret key (free tier available).
    See https://ai.baidu.com/tech/speech/tts
    """

    # Baidu TTS language codes
    LANG_MAP = {
        "zh": "zh", "zh-CN": "zh", "zh-TW": "ct",
        "en": "en", "en-US": "en", "en-GB": "uk",
        "ja": "jp", "ja-JP": "jp",
        "ko": "kor", "ko-KR": "kor",
        "fr": "fra", "de": "de", "es": "spa",
    }

    # Speaker → Baidu 'per' parameter
    SPEAKER_MAP = {
        "alloy": 0, "echo": 1, "fable": 2, "onyx": 3, "nova": 4, "shimmer": 5,
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    }

    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str,
        speaker: str = "1",
        speed: int = 5,
        pitch: int = 5,
        volume: int = 5,
        audio_format: str = "wav",
        timeout: float = 10.0,
        debug: bool = False,
    ):
        super().__init__(timeout=timeout, debug=debug)
        self.api_key = api_key
        self.secret_key = secret_key
        self.speaker = str(speaker)
        self.speed = speed
        self.pitch = pitch
        self.volume = volume
        self.audio_format = audio_format
        self._access_token: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    # -- synthesize --------------------------------------------------------

    async def synthesize(self, text: str, language: str = None) -> bytes:
        if not text or not text.strip():
            return b""

        if self.debug:
            logger.info(f"TTS: '{text}'")

        lan = self._map_language(language)
        per = self.SPEAKER_MAP.get(self.speaker, 1)

        params = {
            "tex": urllib.parse.quote(text, safe=""),
            "tok": await asyncio.to_thread(self._get_access_token),
            "cuid": "voice_assistant",
            "ctp": 1,
            "lan": lan,
            "spd": self.speed,
            "pit": self.pitch,
            "vol": self.volume,
            "per": per,
            "aue": 6,  # WAV format
        }

        resp = await self._client().post(
            url="https://tsn.baidu.com/text2audio",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=urllib.parse.urlencode(params, encoding="utf-8"),
        )

        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            try:
                err = resp.json()
                logger.error(f"Baidu TTS error: {err}")
            except Exception:
                logger.error(f"Baidu TTS unexpected response: {ct}")
            return b""

        return resp.content

    # -- helpers -----------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        """httpx client 懒建(建 SSL 上下文要 ~1s, 别放在构造/事件循环里)。"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._http_client

    async def close(self):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _map_language(self, language: str) -> str:
        """Map a language code to Baidu's 'lan' parameter."""
        if not language:
            return "zh"
        mapped = self.LANG_MAP.get(language) or self.LANG_MAP.get(language.split("-")[0])
        if not mapped:
            logger.warning(f"Unsupported TTS language '{language}', fallback to 'zh'")
            return "zh"
        return mapped

    def _get_access_token(self) -> str:
        """Get or refresh Baidu OAuth access token(同步, 调用方用 to_thread 包住)。"""
        if self._access_token:
            return self._access_token
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }
        resp = httpx.post(url, params=params)
        self._access_token = resp.json().get("access_token", "")
        return self._access_token
