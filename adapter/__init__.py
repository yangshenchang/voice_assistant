"""Audio device adapter — microphone input + speaker output.

Captures mic audio → feeds through VAD → pumps pipeline responses
to the speaker in a background thread.
"""

import asyncio
import logging
import queue
import struct
import threading
import time
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
        mic_queue_size: int = 100,
        cancel_echo: bool = True,
    ):
        self.sts = sts
        self.input_sample_rate = input_sample_rate
        self.input_channels = input_channels
        self.input_chunk_size = input_chunk_size
        self.output_chunk_size = output_chunk_size
        #: 麦克风线程 → 事件循环的队列长度（满则丢最老的一块，上限约 3.2s 音频）
        self.mic_queue_size = mic_queue_size

        # Wire pipeline callbacks to this adapter
        sts.handle_response = self.handle_response
        sts.stop_response = self.stop_response

        # PyAudio
        self._pa = pyaudio.PyAudio()
        self._play_stream: Optional[pyaudio.Stream] = None
        self._play_rate: Optional[int] = None   # 当前播放流的采样率
        self._play_channels: Optional[int] = None

        # Echo control
        self.cancel_echo = cancel_echo
        self._playing = False
        sts.vad.should_mute = (lambda: self._playing) if cancel_echo else (lambda: False)

        # Playback thread
        self._response_queue: queue.Queue = queue.Queue()
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
        """Start the microphone → VAD pipeline. Blocks until cancelled.

        **麦克风读取放在独立线程**（原先 `stream.read()` 是同步阻塞调用、直接跑在
        事件循环线程上，每 32ms 卡一次循环）。实测代价很大：开麦状态下
        `请求→首token` 445→700ms、`token→句` 146→500ms —— LLM 段凭空多 600ms，
        因为 SSE 增量被阻塞读"晚处理"了。线程 + 队列后事件循环不再被卡。
        """
        loop = asyncio.get_running_loop()
        mic_q: asyncio.Queue = asyncio.Queue(maxsize=self.mic_queue_size)
        stop = threading.Event()

        def _offer(data: bytes):
            """在事件循环线程上入队；满了丢最老的一块，避免音频越堆越多。"""
            if mic_q.full():
                try:
                    mic_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                mic_q.put_nowait(data)
            except asyncio.QueueFull:
                pass

        def reader():
            stream = self._pa.open(
                rate=self.input_sample_rate,
                channels=self.input_channels,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=self._input_device_index,
                frames_per_buffer=self.input_chunk_size,
            )
            try:
                while not stop.is_set():
                    data = stream.read(self.input_chunk_size, exception_on_overflow=False)
                    try:
                        loop.call_soon_threadsafe(_offer, data)
                    except RuntimeError:
                        break          # 事件循环已关闭
            except Exception as ex:                                 # noqa: BLE001
                logger.error(f"Mic reader stopped: {ex}")
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:                                   # noqa: BLE001
                    pass

        thread = threading.Thread(target=reader, daemon=True, name="mic-reader")
        thread.start()

        if user_id:
            self.sts.vad.set_session_data(session_id, "user_id", user_id, create=True)
        # 走 pipeline 的音频入口: 顺路喂流式 KWS(唤醒词跟着音频判, 不占链路时间)
        try:
            while True:
                chunk = await mic_q.get()
                await self.sts.feed_audio(chunk, session_id)
        finally:
            stop.set()
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Speaker output (background thread)
    # ------------------------------------------------------------------

    def _player_worker(self):
        """Consume PCM chunks from the queue and play them.

        队列元素:
          (sample_rate, pcm_bytes)  —— 16bit 单声道 PCM, 直接写扬声器
          None                      —— 本轮结束(等设备缓冲放完再解除静音)
        """
        while True:
            item = self._response_queue.get()
            try:
                if item is None:
                    self._finish_utterance()
                    continue

                rate, pcm = item
                if not pcm:
                    continue
                self._playing = True

                stream, play_rate = self._stream_for(rate)
                if play_rate != rate:
                    # 设备不接受这个采样率才走这里(极少见, 见 _stream_for)
                    pcm = self._resample(pcm, rate, play_rate, 2, 1)

                chunk_bytes = self.output_chunk_size * 2
                for offset in range(0, len(pcm), chunk_bytes):
                    chunk = pcm[offset:offset + chunk_bytes]
                    if chunk:
                        stream.write(chunk)

            except Exception as ex:
                logger.error(f"Playback error: {ex}", exc_info=True)
            finally:
                self._response_queue.task_done()

    def _stream_for(self, rate: int):
        """拿到一个能放 *rate* 的输出流。

        优先按音频自己的采样率打开(WASAPI/MME 共享模式下系统会在 C 层重采样),
        只有设备拒绝时才退回设备默认率, 那时才需要自己重采样 —— 之前是无条件
        用设备默认率, 于是每句话都要在播放线程里跑一遍纯 Python 重采样
        (实测 1.5~3s 的句子要 250~625ms), 直接拖慢首字出声。
        """
        if (self._play_stream is not None and self._play_rate == rate
                and self._play_channels == 1):
            return self._play_stream, self._play_rate

        self._close_play_stream()
        try:
            self._play_stream = self._pa.open(
                format=pyaudio.paInt16, channels=1, rate=rate, output=True,
                output_device_index=self._output_device_index,
            )
            self._play_rate = rate
        except Exception as ex:
            logger.warning(f"Cannot open output at {rate}Hz ({ex}); using device rate")
            self._play_stream = self._pa.open(
                format=pyaudio.paInt16, channels=1, rate=self._output_sample_rate,
                output=True, output_device_index=self._output_device_index,
            )
            self._play_rate = self._output_sample_rate
        self._play_channels = 1
        return self._play_stream, self._play_rate

    def _close_play_stream(self):
        if self._play_stream is not None:
            try:
                self._play_stream.stop_stream()
                self._play_stream.close()
            except Exception:
                pass
            self._play_stream = None
            self._play_rate = None
            self._play_channels = None

    def _finish_utterance(self):
        """本轮音频已全部写完: 等设备缓冲真正放完, 再解除麦克风静音。"""
        if self._play_stream is not None:
            try:
                latency = self._play_stream.get_output_latency()
            except Exception:
                latency = 0.2
            time.sleep(min(max(latency, 0.05), 1.5))
        self._playing = not self._response_queue.empty()   # 下一轮可能已经排上了

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
        """Called by the pipeline for each response chunk.

        ``chunk`` 带 PCM 音频 → 入队播放; ``final`` → 本轮音频结束。
        """
        if response.type == "chunk" and response.audio_data:
            meta = response.metadata or {}
            rate = meta.get("sample_rate") or getattr(self.sts.tts, "sample_rate", 16000)
            self._response_queue.put((rate, response.audio_data))
        elif response.type == "final":
            self._response_queue.put(None)

    async def stop_response(self, session_id: str = None, context_id: str = None):
        """Stop playback and clear the queue.

        没在放音、队列也空、输出流也没开着时 **直接返回**：这种"什么都没在跑"的
        情况占多数轮次，而 `_close_play_stream()` 会真的去关一次音频设备，下一轮
        播放时再开一次 —— 在无线音箱上这一关一开是几百毫秒级，且恰好落在
        "用户说完话 → 发 LLM 请求" 的关键路径上（实测口径里的"请求前"开销）。
        """
        if self._play_stream is None and not self._playing and self._response_queue.empty():
            return
        self._close_play_stream()
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
                self._response_queue.task_done()
            except queue.Empty:
                break
        self._playing = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self._close_play_stream()
        self._pa.terminate()
