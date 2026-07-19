"""Voice Activity Detection (VAD) — detects speech segments from an audio stream."""

import asyncio
import logging
import math
import struct
from abc import ABC, abstractmethod
from collections import deque
from typing import AsyncGenerator, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class SpeechDetector(ABC):
    """Abstract base for speech detectors."""

    def __init__(self, *, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._on_speech_detected: Callable = self._default_handler
        self.should_mute: Callable[[], bool] = lambda: False

    def on_speech_detected(self, func):
        """Decorator / setter for the speech-detected callback."""
        self._on_speech_detected = func
        return func

    async def _default_handler(self, data: bytes, duration: float, session_id: str):
        logger.info(f"Speech detected: {duration:.1f}s (session={session_id})")

    @abstractmethod
    async def process_samples(self, samples: bytes, session_id: str = None):
        """Feed raw audio samples."""

    @abstractmethod
    async def process_stream(self, stream: AsyncGenerator[bytes, None], session_id: str = None):
        """Consume an async generator of audio chunks."""

    @abstractmethod
    async def finalize_session(self, session_id: str):
        """Clean up a session."""


# ---------------------------------------------------------------------------
# Standard amplitude-threshold detector
# ---------------------------------------------------------------------------

class _RecordingSession:
    """Per-session VAD state."""

    __slots__ = ("session_id", "is_recording", "buffer", "silence_duration",
                 "record_duration", "preroll_buffer", "amplitude_threshold", "data")

    def __init__(self, session_id: str, preroll_frames: int = 5):
        self.session_id = session_id
        self.is_recording = False
        self.buffer = bytearray()
        self.silence_duration = 0.0
        self.record_duration = 0.0
        self.preroll_buffer = deque(maxlen=preroll_frames)
        self.amplitude_threshold = 0.0
        self.data: dict = {}

    def reset(self):
        self.buffer.clear()
        self.is_recording = False
        self.silence_duration = 0.0
        self.record_duration = 0.0


class StandardSpeechDetector(SpeechDetector):
    """Amplitude-threshold based VAD.

    Parameters
    ----------
    volume_db_threshold:
        dB threshold below which audio is considered silence (default -40).
    silence_duration_threshold:
        Seconds of continuous silence to end a speech segment (default 0.5).
    max_duration:
        Maximum recording duration in seconds (default 10).
    min_duration:
        Minimum recording duration in seconds (default 0.2).
    sample_rate:
        Audio sample rate in Hz (default 16000).
    channels:
        Number of audio channels (default 1).
    preroll_buffer_count:
        Number of frames to keep before speech onset (default 5).
    """

    def __init__(
        self,
        *,
        volume_db_threshold: float = -40.0,
        silence_duration_threshold: float = 0.5,
        max_duration: float = 10.0,
        min_duration: float = 0.2,
        sample_rate: int = 16000,
        channels: int = 1,
        preroll_buffer_count: int = 5,
        debug: bool = False,
    ):
        super().__init__(sample_rate=sample_rate)
        self._volume_db_threshold = volume_db_threshold
        self.amplitude_threshold = self._db_to_amplitude(volume_db_threshold)
        self.silence_duration_threshold = silence_duration_threshold
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.channels = channels
        self.preroll_buffer_count = preroll_buffer_count
        self.debug = debug
        self._sessions: Dict[str, _RecordingSession] = {}

    # -- properties --------------------------------------------------------

    @property
    def volume_db_threshold(self) -> float:
        return self._volume_db_threshold

    @volume_db_threshold.setter
    def volume_db_threshold(self, value: float):
        self._volume_db_threshold = value
        self.amplitude_threshold = self._db_to_amplitude(value)

    @staticmethod
    def _db_to_amplitude(db: float) -> float:
        return 32767 * (10 ** (db / 20.0))

    # -- public API --------------------------------------------------------

    async def process_samples(self, samples: bytes, session_id: str):
        if self.should_mute():
            self._get_session(session_id).reset()
            return

        session = self._get_session(session_id)
        session.preroll_buffer.append(samples)

        max_amp = float(max(abs(s[0]) for s in struct.iter_unpack("<h", samples)))
        sample_dur = (len(samples) / 2) / (self.sample_rate * self.channels)

        if not session.is_recording:
            if max_amp > session.amplitude_threshold:
                session.reset()
                session.is_recording = True
                for f in session.preroll_buffer:
                    session.buffer.extend(f)
                session.buffer.extend(samples)
                session.record_duration += sample_dur
        else:
            session.buffer.extend(samples)
            session.record_duration += sample_dur

            if max_amp > session.amplitude_threshold:
                session.silence_duration = 0
            else:
                session.silence_duration += sample_dur

            if session.silence_duration >= self.silence_duration_threshold:
                recorded_dur = session.record_duration - session.silence_duration
                if recorded_dur < self.min_duration:
                    if self.debug:
                        logger.info(f"Too short: {recorded_dur:.2f}s")
                else:
                    if self.debug:
                        logger.info(f"Segment: {recorded_dur:.2f}s")
                    data = bytes(session.buffer)
                    asyncio.create_task(
                        self._on_speech_detected(data, recorded_dur, session.session_id)
                    )
                session.reset()

            elif session.record_duration >= self.max_duration:
                if self.debug:
                    logger.info(f"Too long: {session.record_duration:.2f}s")
                session.reset()

    async def process_stream(self, stream: AsyncGenerator[bytes, None], session_id: str):
        logger.info("VAD stream started")
        async for chunk in stream:
            if not chunk:
                break
            await self.process_samples(chunk, session_id)
            await asyncio.sleep(0.0001)
        self._sessions.pop(session_id, None)
        logger.info("VAD stream ended")

    async def finalize_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    # -- session helpers ---------------------------------------------------

    def _get_session(self, session_id: str) -> _RecordingSession:
        if session_id not in self._sessions:
            s = _RecordingSession(session_id, self.preroll_buffer_count)
            s.amplitude_threshold = self.amplitude_threshold
            self._sessions[session_id] = s
        return self._sessions[session_id]

    def get_session_data(self, session_id: str, key: str):
        s = self._sessions.get(session_id)
        return s.data.get(key) if s else None

    def set_session_data(self, session_id: str, key: str, value, create: bool = False):
        if create:
            s = self._get_session(session_id)
        else:
            s = self._sessions.get(session_id)
        if s:
            s.data[key] = value
