"""Audio device adapter — microphone input + speaker output.

Captures mic audio → feeds through VAD → pumps pipeline responses
to the speaker in a background thread.
"""

import asyncio
import io
import logging
import queue
import struct
import threading
import wave
from typing import Optional

import pyaudio

from ..models import STSResponse
from ..pipeline import VoiceAssistant

logger = logging.getLogger(__name__)


class AudioDeviceAdapter:
    """Wires the ``VoiceAssistant`` pipeline to physical audio I/O.

    Parameters
    ----------
    sts:
        The voice assistant pipeline instance.
    input_sample_rate:
        Mic sample rate (Hz). Default 16000.
    input_channels:
        Mic channels (1 = mono).
    input_chunk_size:
        Frames per mic read.
    output_chunk_size:
        Frames per speaker write.
    cancel_echo:
        If True, mute the VAD while the speaker is playing (echo suppression).
    """

    def __init__(
        self,
        sts: "VoiceAssistant",
        *,
        input_sample_rate: int = 16000,
        input_channels: int = 1,
        input_chunk_size: int = 512,
        output_chunk_size: int = 1024,
        cancel_echo: bool = True,
    ):
        self.sts = sts
        self.input_sample_rate = input_sample_rate
        self.input_channels = input_channels
        self.input_chunk_size = input_chunk_size
        self.output_chunk_size = output_chunk_size

        # Wire pipeline callbacks to this adapter
        sts.handle_response = self.handle_response
        sts.stop_response = self.stop_response

        # PyAudio
        self._pa = pyaudio.PyAudio()
        self._play_stream: Optional[pyaudio.Stream] = None
        self._wave_params: Optional[tuple] = None

        # Echo control
        self.cancel_echo = cancel_echo
        self._playing = False
        sts.vad.should_mute = (lambda: self._playing) if cancel_echo else (lambda: False)

        # Playback thread
        self._response_queue: queue.Queue[bytes] = queue.Queue()
        self._playback_thread = threading.Thread(
            target=self._player_worker, daemon=True
        )
        self._playback_thread.start()

        # Discover devices
        self._input_device_index = self._find_device("Microphone", "input")
        self._output_device_index, self._output_sample_rate = self._find_output_device()

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    def _find_device(self, keyword: str, direction: str) -> Optional[int]:
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            name = dev.get("name", "")
            # Only consider devices that support input channels
            if direction == "input" and dev.get("maxInputChannels", 0) <= 0:
                continue
            if direction == "output" and dev.get("maxOutputChannels", 0) <= 0:
                continue
            if keyword.lower() in name.lower():
                print(f"[adapter] {direction}: [{i}] {name}")
                return i
        # Fallback to system default
        try:
            if direction == "input":
                info = self._pa.get_default_input_device_info()
            else:
                info = self._pa.get_default_output_device_info()
            print(f"[adapter] {direction} (default): [{info['index']}] {info['name']}")
            return info["index"]
        except Exception:
            raise RuntimeError(f"No {direction} device found")

    def _find_output_device(self) -> tuple:
        try:
            info = self._pa.get_default_output_device_info()
            idx = info.get("index")
            rate = int(info.get("defaultSampleRate", 48000))
            print(f"[adapter] output: [{idx}] {info['name']} @ {rate}Hz")
            return idx, rate
        except Exception:
            print("[adapter] output: using default 48000Hz")
            return None, 48000

    # ------------------------------------------------------------------
    # Mic input
    # ------------------------------------------------------------------

    async def start_listening(self, session_id: str, user_id: str = None):
        """Start the microphone → VAD pipeline. Blocks until cancelled."""

        async def mic_stream():
            stream = self._pa.open(
                rate=self.input_sample_rate,
                channels=self.input_channels,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=self._input_device_index,
                frames_per_buffer=self.input_chunk_size,
            )
            try:
                while True:
                    yield stream.read(self.input_chunk_size, exception_on_overflow=False)
                    await asyncio.sleep(0.0001)
            finally:
                stream.stop_stream()
                stream.close()

        if user_id:
            self.sts.vad.set_session_data(session_id, "user_id", user_id, create=True)
        await self.sts.vad.process_stream(mic_stream(), session_id)

    # ------------------------------------------------------------------
    # Speaker output (background thread)
    # ------------------------------------------------------------------

    def _player_worker(self):
        """Consume audio chunks from the queue and play them."""
        while True:
            try:
                audio = self._response_queue.get()
                self._playing = True

                with wave.open(io.BytesIO(audio), "rb") as wf:
                    rate, width, channels = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
                    playback_rate = self._output_sample_rate if rate != self._output_sample_rate else rate

                    stream_key = (width, channels, playback_rate)
                    if not self._play_stream or self._wave_params != stream_key:
                        if self._play_stream:
                            try:
                                self._play_stream.close()
                            except Exception:
                                pass
                        self._wave_params = stream_key
                        self._play_stream = self._pa.open(
                            format=self._pa.get_format_from_width(width),
                            channels=channels,
                            rate=playback_rate,
                            output=True,
                            output_device_index=self._output_device_index,
                        )

                    pcm = wf.readframes(wf.getnframes())
                    if rate != playback_rate:
                        pcm = self._resample(pcm, rate, playback_rate, width, channels)

                    chunk_bytes = self.output_chunk_size * width * channels
                    for offset in range(0, len(pcm), chunk_bytes):
                        chunk = pcm[offset:offset + chunk_bytes]
                        if chunk:
                            self._play_stream.write(chunk)

            except Exception as ex:
                logger.error(f"Playback error: {ex}", exc_info=True)
            finally:
                self._playing = not self._response_queue.empty()
                self._response_queue.task_done()

    @staticmethod
    def _resample(data: bytes, orig: int, target: int, width: int, channels: int) -> bytes:
        """Simple linear resampling for 16-bit PCM."""
        if orig == target or not data:
            return data
        fmt = "<h"
        samples = [s[0] for s in struct.iter_unpack(fmt, data)]
        ratio = orig / target
        new_len = max(1, int(len(samples) / ratio))
        out = bytearray()
        for i in range(new_len):
            pos = i * ratio
            idx = int(pos)
            frac = pos - idx
            v = samples[idx]
            if idx + 1 < len(samples):
                v = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            out.extend(struct.pack(fmt, max(-32768, min(32767, v))))
        return bytes(out)

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    async def handle_response(self, response: STSResponse):
        """Called by the pipeline for each response chunk."""
        if response.type == "chunk" and response.audio_data:
            self._response_queue.put(response.audio_data)

    async def stop_response(self, session_id: str = None, context_id: str = None):
        """Immediately stop playback and clear the queue."""
        if self._play_stream:
            try:
                self._play_stream.stop_stream()
                self._play_stream.close()
                self._play_stream = None
                self._wave_params = None
            except Exception:
                pass
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
                self._response_queue.task_done()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self.stop_response()
        self._pa.terminate()
