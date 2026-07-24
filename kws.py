"""Local keyword spotting (KWS) using sherpa-onnx.

Provides CPU-based wake-word detection that runs entirely offline,
eliminating the need to stream all audio to cloud ASR for wake-word checks.

The model uses BPE tokenization of pinyin. Each Chinese character is
converted to pinyin with tone marks (e.g., "xiǎo"), then split into
initial ("x") and final ("iǎo") BPE tokens.
"""

import logging
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
        CPU threads for the ONNX runtime (default 2).
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
        num_threads: int = 2,
        sample_rate: int = 16000,
        keywords_threshold: float = 0.25,
        provider: str = "cpu",
        debug: bool = False,
    ):
        self._sample_rate = sample_rate
        self._keywords = keywords or []
        self._debug = debug

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
