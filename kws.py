"""Local keyword spotting (KWS) using sherpa-onnx.

Provides CPU-based wake-word detection that runs entirely offline,
eliminating the need to stream all audio to cloud ASR for wake-word checks.

The model uses BPE tokenization of pinyin. Each Chinese character is
converted to pinyin with tone marks (e.g., "xiǎo"), then split into
initial ("x") and final ("iǎo") BPE tokens.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)

# Default model directory (extracted from the tarball)
_MODEL_DIR = (
    Path(__file__).parent
    / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
)

# Chinese pinyin initials, ordered longest-first for greedy matching
_PINYIN_INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s", "y", "w",
]


class KeywordWakeDetector:
    """Local keyword-spotting wake-word detector backed by sherpa-onnx.

    Runs a streaming transducer model on CPU to detect keywords in real-time
    from 16-bit PCM audio without any cloud dependency.

    Parameters
    ----------
    model_dir:
        Path to the extracted sherpa-onnx KWS model directory.
    keywords:
        List of Chinese keyword strings (e.g. ``["小牛"]``). These are
        converted to BPE tokens and passed as inline keywords to the spotter.
    num_threads:
        onnxruntime intra-op 线程数, 默认 1。这个 3.3M 的小模型用不到多线程, 而且
        onnxruntime 的线程池会在两次 decode 之间忙等自旋 —— 实时流式喂音频实测:
        =1 占 8% 单核, =2 占 74%, =4 占 147%, 而单次 decode 耗时几乎一样
        (58 / 64 / 56 ms), 所以 1 线程既省 CPU 又不慢。
    sample_rate:
        Input audio sample rate in Hz (default 16000).
    keywords_threshold:
        Confidence threshold for keyword detection (default 0.25).
    provider:
        ONNX execution provider (default ``"cpu"``).
    debug:
        Enable verbose logging.
    """

    def __init__(
        self,
        *,
        model_dir: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        num_threads: int = 1,
        sample_rate: int = 16000,
        keywords_threshold: float = 0.25,
        provider: str = "cpu",
        stream_queue_size: int = 100,
        debug: bool = False,
    ):
        self._sample_rate = sample_rate
        self._keywords = keywords or []
        self._debug = debug
        # detect() 是整段批量判, 可能从线程池里并发进来, 而 sherpa spotter 是共享的
        # 原生对象 → 用锁串行化。(流式路径只在 _stream_worker 单线程里碰 spotter。)
        self._lock = threading.Lock()
        # 流式状态
        self._streams: dict = {}
        self._hits: dict = {}
        self._queue: "queue.Queue" = queue.Queue(maxsize=stream_queue_size)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()

        model_path = Path(model_dir) if model_dir else _MODEL_DIR
        if not model_path.exists():
            raise FileNotFoundError(
                f"KWS model directory not found: {model_path}"
            )

        # Use int8 encoder/joiner for speed, float decoder for accuracy
        encoder = str(model_path / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx")
        decoder = str(model_path / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        joiner = str(model_path / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx")
        tokens = str(model_path / "tokens.txt")
        keywords_file = str(model_path / "keywords.txt")

        # Pre-load the set of valid BPE tokens for validation
        self._valid_tokens = self._load_token_set(tokens)

        # Build inline keyword string from Chinese keywords
        self._inline_keywords = self._build_keyword_string(self._keywords)

        if self._debug:
            logger.info(f"KWS model: {model_path}")
            logger.info(f"KWS keywords: {self._keywords}")
            if self._inline_keywords:
                logger.info(f"KWS inline tokens: {self._inline_keywords}")

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_file,
            num_threads=num_threads,
            sample_rate=sample_rate,
            keywords_threshold=keywords_threshold,
            provider=provider,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Streaming API (推荐用法)
    #
    # 边收麦克风音频边解码, 唤醒判定跟着音频走, 完全不占用对话链路的时间。
    # 实测这个模型一个 chunk = 320ms, 单次 decode_stream ≈ 52ms(16% 单核),
    # 所以解码放在一个后台线程里, feed() 本身只是入队(微秒级), 可以放心在
    # 事件循环里每个音频块调一次。
    # ------------------------------------------------------------------

    def feed(self, session_id: str, pcm_bytes: bytes):
        """把一个音频块(如 512 帧/32ms)丢进流式队列。不阻塞, 不等待结果。"""
        if not pcm_bytes:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait((session_id, pcm_bytes))
        except queue.Full:
            # 解码跟不上(正常不会发生): 丢掉最老的一块, 保证音频不会越堆越多
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait((session_id, pcm_bytes))
            except (queue.Empty, queue.Full):
                pass

    def poll(self, session_id: str) -> Optional[str]:
        """取走该会话最近一次命中的唤醒词(没有则 None)。非阻塞。"""
        return self._hits.pop(session_id, None)

    def has_stream(self, session_id: str) -> bool:
        """该会话是否已经建立流式状态(说明音频是通过 feed() 进来的)。"""
        return session_id in self._streams

    def reset_stream(self, session_id: str):
        """丢掉该会话的流式状态(会话睡着时调, 免得残留上下文影响下一次唤醒)。"""
        stream = self._streams.pop(session_id, None)
        if stream is not None:
            try:
                self._spotter.reset_stream(stream)
            except Exception:
                pass

    def close(self):
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        self._streams.clear()
        self._hits.clear()

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._stream_worker, daemon=True, name="kws-stream"
            )
            self._worker.start()

    def _stream_worker(self):
        """后台线程: 出队音频 → 喂给该会话的流 → 命中就记下来等 poll() 取。"""
        while not self._stop.is_set():
            try:
                session_id, pcm_bytes = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                samples = (
                    np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                )
                stream = self._streams.get(session_id)
                if stream is None:
                    stream = self._spotter.create_stream(self._inline_keywords or None)
                    self._streams[session_id] = stream

                stream.accept_waveform(self._sample_rate, samples)
                while self._spotter.is_ready(stream):
                    self._spotter.decode_stream(stream)
                    result = self._spotter.get_result(stream)
                    if result:
                        if self._debug:
                            logger.info(f"KWS detected (streaming): '{result}'")
                        self._spotter.reset_stream(stream)
                        self._hits[session_id] = result
                        break
            except Exception as e:                       # noqa: BLE001
                logger.error(f"KWS stream error: {e}")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Batch API (整段判一次, 用于没有走 feed() 的调用方)
    # ------------------------------------------------------------------

    def detect(self, pcm_bytes: bytes) -> Optional[str]:
        """Run keyword detection on a chunk of 16-bit PCM audio.

        Parameters
        ----------
        pcm_bytes:
            Raw 16-bit, mono PCM audio data.

        Returns
        -------
        The detected keyword string (e.g. ``"小牛"``), or ``None`` if no
        keyword was detected.
        """
        if not pcm_bytes:
            return None

        with self._lock:
            # Convert int16 PCM → float32 numpy array
            samples = (
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Create a fresh stream with our custom inline keywords
            stream = self._spotter.create_stream(self._inline_keywords or None)
            stream.accept_waveform(self._sample_rate, samples)

            # Tail padding: 0.66s of silence to flush the model
            tail = np.zeros(int(0.66 * self._sample_rate), dtype=np.float32)
            stream.accept_waveform(self._sample_rate, tail)
            stream.input_finished()

            # Decode until done
            while self._spotter.is_ready(stream):
                self._spotter.decode_stream(stream)
                result = self._spotter.get_result(stream)
                if result:
                    if self._debug:
                        logger.info(f"KWS detected: '{result}'")
                    self._spotter.reset_stream(stream)
                    return result

        return None

    # ------------------------------------------------------------------
    # Pinyin → BPE conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _load_token_set(tokens_path: str) -> set:
        """Load the set of valid BPE token strings from tokens.txt."""
        tokens = set()
        with open(tokens_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "token id" or "token"
                parts = line.split()
                if parts:
                    tokens.add(parts[0])
        return tokens

    @staticmethod
    def _split_pinyin(pinyin: str) -> tuple:
        """Split a pinyin syllable into (initial, final) pair.

        Examples
        --------
        "xiǎo" → ("x", "iǎo")
        "niú"  → ("n", "iú")
        "ài"   → ("", "ài")       # zero initial
        "ān"   → ("", "ān")
        """
        for initial in _PINYIN_INITIALS:
            if pinyin.startswith(initial):
                final = pinyin[len(initial):]
                if final:
                    return initial, final
                # The whole string is the initial? That shouldn't happen
                # for valid pinyin, but handle gracefully
                return initial, ""
        # Zero initial (e.g. "ài", "ěr", "ǒu")
        return "", pinyin

    def _pinyin_to_bpe(self, pinyin: str) -> Optional[str]:
        """Convert a single pinyin syllable (with tone marks) to BPE tokens.

        Returns a space-separated string like ``"x iǎo"`` or ``"ài"``,
        or ``None`` if the syllable can't be decomposed into known tokens.
        """
        initial, final = self._split_pinyin(pinyin)

        if initial and initial not in self._valid_tokens:
            if self._debug:
                logger.warning(
                    f"BPE token not found for initial '{initial}' (from '{pinyin}')"
                )
            return None

        if final and final not in self._valid_tokens:
            if self._debug:
                logger.warning(
                    f"BPE token not found for final '{final}' (from '{pinyin}')"
                )
            return None

        if initial and final:
            return f"{initial} {final}"
        elif final:
            # Zero-initial syllable
            return final
        else:
            return None

    def _chars_to_bpe(self, keyword: str) -> Optional[str]:
        """Convert a Chinese keyword to BPE inline keyword format.

        Returns a string like ``"x iǎo n iú @小牛"`` or ``None`` on failure.
        """
        try:
            from pypinyin import lazy_pinyin, Style
            # Style.TONE gives tone marks: "xiǎo", "niú"
            pinyins = lazy_pinyin(keyword, style=Style.TONE)
        except ImportError:
            logger.warning(
                "pypinyin not installed — cannot convert keywords to BPE. "
                "Install with: pip install pypinyin"
            )
            return None

        bpe_parts = []
        for py in pinyins:
            bpe = self._pinyin_to_bpe(py)
            if bpe is None:
                logger.warning(
                    f"Cannot convert pinyin '{py}' in '{keyword}' to BPE tokens"
                )
                return None
            bpe_parts.append(bpe)

        if not bpe_parts:
            return None

        return " ".join(bpe_parts) + f" @{keyword}"

    def _build_keyword_string(self, keywords: List[str]) -> str:
        """Build inline keyword string from a list of Chinese keywords.

        Multiple keywords are joined with ``/`` as the separator.
        """
        parts = []
        for kw in keywords:
            bpe = self._chars_to_bpe(kw)
            if bpe:
                parts.append(bpe)
            else:
                logger.warning(
                    f"Cannot convert keyword '{kw}' to BPE tokens, skipping"
                )
        return "/".join(parts) if parts else ""
