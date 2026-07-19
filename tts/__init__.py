"""Text-to-Speech engines: Baidu TTS and Alibaba Cloud CosyVoice."""

import asyncio
import logging
import os
import urllib.parse
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import httpx
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SpeechSynthesizer(ABC):
    """Abstract base for text-to-speech engines."""

    def __init__(self, *, timeout: float = 10.0, debug: bool = False):
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        self.debug = debug

    @abstractmethod
    async def synthesize(self, text: str, language: str = None) -> bytes:
        """Synthesize text to audio bytes. Returns empty bytes on failure."""

    async def close(self):
        await self.http_client.aclose()


# ---------------------------------------------------------------------------
# CosyVoice TTS (Alibaba Cloud DashScope) — Recommended
# ---------------------------------------------------------------------------

class CosyVoiceSpeechSynthesizer(SpeechSynthesizer):
    """Text-to-speech via Alibaba Cloud DashScope CosyVoice.

    Uses CosyVoice v3-flash by default — the fastest model optimized for
    real-time voice interaction with natural human-like prosody and pausing.

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
        super().__init__(timeout=timeout, debug=debug)
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
        self.sample_rate = sample_rate

    # -- synthesize --------------------------------------------------------

    async def synthesize(self, text: str, language: str = None) -> bytes:
        if not text or not text.strip():
            return b""

        if self.debug:
            logger.info(f"CosyVoice TTS: '{text}'")

        language_hints = self._map_language(language)

        try:
            result = await asyncio.to_thread(
                self._call_http_tts, text, language_hints
            )
            return result
        except Exception as e:
            logger.error(f"CosyVoice TTS error: {e}")
            return b""

    def _call_http_tts(self, text: str, language_hints: List[str]) -> bytes:
        """Synchronous HTTP TTS call (runs in thread pool)."""
        from dashscope.audio.http_tts.http_speech_synthesizer import (
            HttpSpeechSynthesizer,
        )

        stream_result = HttpSpeechSynthesizer.call(
            model=self.model,
            text=text,
            voice=self.voice,
            audio_format="wav",
            sample_rate=self.sample_rate,
            stream=True,
            api_key=self.api_key,
            speech_rate=self.speech_rate,
            pitch_rate=self.pitch_rate,
            volume=self.volume,
            language_hints=language_hints,
        )

        audio_chunks = []
        for chunk in stream_result:
            if chunk.audio_data and not getattr(chunk, "audio_url", None):
                audio_chunks.append(chunk.audio_data)

        return b"".join(audio_chunks)

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
            "tok": self._get_access_token(),
            "cuid": "voice_assistant",
            "ctp": 1,
            "lan": lan,
            "spd": self.speed,
            "pit": self.pitch,
            "vol": self.volume,
            "per": per,
            "aue": 6,  # WAV format
        }

        resp = await self.http_client.post(
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
        """Get or refresh Baidu OAuth access token."""
        if self._access_token:
            return self._access_token
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }
        resp = requests.post(url, params=params)
        self._access_token = resp.json().get("access_token", "")
        return self._access_token
