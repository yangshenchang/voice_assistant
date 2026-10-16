"""Speech-to-Text via Alibaba Cloud DashScope (qwen3-asr-flash)."""

import asyncio
import base64
import io
import logging
import os
import wave
from abc import ABC, abstractmethod
from typing import List, Optional

import dashscope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SpeechRecognizer(ABC):
    """Abstract base for speech-to-text engines."""

    def __init__(self, *, language: str = None, debug: bool = False):
        self.language = language
        self.debug = debug

    @abstractmethod
    async def transcribe(self, pcm_data: bytes) -> str:
        """Transcribe PCM audio to text. Returns empty string on failure."""


class NullSpeechRecognizer(SpeechRecognizer):
    """占位识别器：**文本模式专用**，不持有任何凭据、不联网。

    为什么需要它：``AliyunASRSpeechRecognizer`` 在**构造时**就要求 API key，
    于是"只想用文本模式跑链路逻辑"的场景会因为缺 DASHSCOPE key 直接崩在启动
    阶段 —— 明明整条文本路径一行 STT 都不经过。

    这个类让 pipeline 能在没有 STT 凭据的情况下构造成功；而一旦真有人把音频喂
    进来，它会**明确抛错**而不是假装返回空串 —— 静默返回空串会让"文本模式误接
    麦克风"这类接线错误变得极难定位。
    """

    def __init__(self, *, debug: bool = False):
        super().__init__(language=None, debug=debug)

    async def transcribe(self, pcm_data: bytes) -> str:
        raise RuntimeError(
            "当前使用 NullSpeechRecognizer（文本模式）：没有可用的语音识别后端。"
            " 请配置 DASHSCOPE_API_KEY，或改用文本输入（run.py --ask / --text）。"
        )


# ---------------------------------------------------------------------------
# Aliyun DashScope ASR
# ---------------------------------------------------------------------------

class AliyunASRSpeechRecognizer(SpeechRecognizer):
    """Speech recognition via Alibaba DashScope ``qwen3-asr-flash``.

    Requires ``DASHSCOPE_API_KEY`` env var or explicit *api_key*.
    """

    def __init__(
        self,
        api_key: str = None,
        sample_rate: int = 16000,
        language: str = "zh",
        model: str = "qwen3-asr-flash",
        max_retries: int = 2,
        *,
        debug: bool = False,
    ):
        super().__init__(language=language, debug=debug)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY env var or api_key param is required")
        dashscope.api_key = self.api_key
        self.sample_rate = sample_rate
        self.model = model
        self.max_retries = max_retries

    # -- transcribe --------------------------------------------------------

    async def transcribe(self, pcm_data: bytes) -> str:
        if not pcm_data:
            return ""

        audio_url = self._pcm_to_data_url(pcm_data)

        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.to_thread(self._call_api, audio_url)
                if result:
                    return result
                if attempt < self.max_retries:
                    logger.warning(f"ASR empty (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(0.3)
            except Exception as ex:
                logger.error(f"ASR error (attempt {attempt + 1}/{self.max_retries + 1}): {ex}")
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        logger.error("All ASR attempts failed")
        return ""

    # -- helpers -----------------------------------------------------------

    def _pcm_to_data_url(self, pcm: bytes) -> str:
        """Convert raw PCM to a base64 data-URL."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    def _call_api(self, audio_url: str) -> str:
        """Synchronous DashScope call (wrapped in ``asyncio.to_thread``)."""
        resp = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=[{"role": "user", "content": [{"audio": audio_url}]}],
            asr_options={"language": self.language, "enable_lid": False, "enable_itn": True},
        )

        if resp.status_code != 200:
            logger.error(f"ASR API status={resp.status_code} code={resp.code} msg={resp.message}")
            return ""

        try:
            text = resp.output.choices[0].message.content[0].get("text", "")
        except (AttributeError, IndexError, KeyError):
            return ""

        if text and self.debug:
            logger.info(f"ASR: '{text}'")
        return text
