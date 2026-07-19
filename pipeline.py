"""Core voice assistant pipeline: VAD → STT → LLM(nanobot) → TTS → speaker.

Orchestrates the full speech-to-speech flow with wake-word detection,
barge-in interruption, and context timeout management.
"""

import asyncio
import logging
import os
import traceback
from time import time
from typing import AsyncGenerator, Callable, Dict, List, Tuple
from uuid import uuid4

from .models import STSRequest, STSResponse
from .vad import SpeechDetector, StandardSpeechDetector
from .stt import AliyunASRSpeechRecognizer, SpeechRecognizer
from .llm import LLMService, LLMResponse, NanobotWebSocketService
from .tts import BaiduSpeechSynthesizer, CosyVoiceSpeechSynthesizer, SpeechSynthesizer

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Speech-to-Speech voice assistant pipeline.

    Quick start::

        sts = VoiceAssistant(
            stt_api_key="...",
            tts_api_key="...",
            tts_secret_key="...",
            wakewords=["你好"],
            debug=True,
        )
        adapter = AudioDeviceAdapter(sts)
        await adapter.start_listening("session_1")

    Parameters
    ----------
    stt_api_key:
        Aliyun DashScope API key (defaults to ``DASHSCOPE_API_KEY`` env var).
    stt_language:
        ASR language code (default ``"zh"``).
    tts_api_key:
        DashScope API key for CosyVoice TTS, or Baidu TTS API key.
        Defaults to ``DASHSCOPE_API_KEY`` env var for CosyVoice.
    tts_secret_key:
        Baidu TTS secret key (only needed for Baidu TTS).
    tts_speaker:
        Baidu TTS speaker id (default ``"1"``). Not used by CosyVoice.
    tts_voice:
        CosyVoice voice id (default ``"longanlingxi"``).
    tts_model:
        CosyVoice model name (default ``"cosyvoice-v3-flash"``).
    tts_provider:
        TTS provider: ``"cosyvoice"`` (default) or ``"baidu"``.
    nanobot_ws_url:
        nanobot WebSocket endpoint (default ``ws://127.0.0.1:8765/``).
    nanobot_token:
        Shared secret for nanobot auth.
    llm_system_prompt:
        System prompt prepended to each new chat.
    vad_volume_threshold:
        VAD dB threshold (default -40).
    vad_silence_threshold:
        Seconds of silence to end a speech segment (default 0.5).
    wakewords:
        List of wake words. No wake words = always awake.
    end_words:
        List of phrases that end the awake session after the current response.
    awake_timeout:
        Seconds before the awake state expires (default 60).
    context_timeout:
        Seconds before a context is considered expired (default 3600).
    barge_in_keywords:
        Phrases that allow the user to interrupt the AI mid-speech.
    debug:
        Enable verbose logging.
    """

    # Default Chinese barge-in keywords
    DEFAULT_BARGE_IN = [
        "慢着", "等一下", "等等", "稍等", "稍等一下",
        "停", "停下", "停一下", "别说了", "先别", "别急",
        "打断", "打断一下", "插一句", "且慢",
        "不对", "不是", "错了", "搞错了", "说错了",
    ]

    def __init__(
        self,
        *,
        # STT
        stt: SpeechRecognizer = None,
        stt_api_key: str = None,
        stt_language: str = "zh",
        # TTS
        tts: SpeechSynthesizer = None,
        tts_api_key: str = None,
        tts_secret_key: str = None,
        tts_speaker: str = "1",
        tts_voice: str = "longanhuan",
        tts_model: str = "cosyvoice-v3-flash",
        tts_provider: str = "cosyvoice",
        # LLM (nanobot)
        llm: LLMService = None,
        nanobot_ws_url: str = "ws://127.0.0.1:8765/",
        nanobot_token: str = "litests-shared-secret",
        llm_system_prompt: str = None,
        llm_model: str = "deepseek-v4-flash",
        # VAD
        vad: SpeechDetector = None,
        vad_volume_threshold: float = -40.0,
        vad_silence_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        # Wake / context
        wakewords: List[str] = None,
        end_words: List[str] = None,
        awake_timeout: float = 60.0,
        context_timeout: float = 3600.0,
        # Barge-in
        barge_in_keywords: List[str] = None,
        # Misc
        debug: bool = False,
    ):
        self.debug = debug
        self._setup_logging()

        # --- VAD ---------------------------------------------------------
        self.vad = vad or StandardSpeechDetector(
            volume_db_threshold=vad_volume_threshold,
            silence_duration_threshold=vad_silence_threshold,
            sample_rate=vad_sample_rate,
            debug=debug,
        )

        @self.vad.on_speech_detected
        async def _on_speech(data: bytes, dur: float, sid: str):
            async for resp in self.invoke(STSRequest(
                session_id=sid,
                user_id=self.vad.get_session_data(sid, "user_id"),
                context_id=self.vad.get_session_data(sid, "context_id"),
                audio_data=data,
                audio_duration=dur,
            )):
                if resp.type == "start":
                    self.vad.set_session_data(sid, "context_id", resp.context_id)
                await self.handle_response(resp)

        # --- STT ---------------------------------------------------------
        if stt:
            self.stt = stt
        else:
            self.stt = AliyunASRSpeechRecognizer(
                api_key=stt_api_key or os.getenv("DASHSCOPE_API_KEY"),
                language=stt_language,
                debug=debug,
            )

        # --- LLM (nanobot) ------------------------------------------------
        if llm:
            self.llm = llm
        else:
            self.llm = NanobotWebSocketService(
                ws_url=nanobot_ws_url,
                token=nanobot_token,
                system_prompt=llm_system_prompt,
                model=llm_model,
                debug=debug,
            )

        # --- TTS ---------------------------------------------------------
        if tts:
            self.tts = tts
        elif tts_provider == "cosyvoice":
            self.tts = CosyVoiceSpeechSynthesizer(
                api_key=tts_api_key,
                model=tts_model,
                voice=tts_voice,
                debug=debug,
            )
        else:
            self.tts = BaiduSpeechSynthesizer(
                api_key=tts_api_key,
                secret_key=tts_secret_key,
                speaker=tts_speaker,
                debug=debug,
            )

        # --- Wake / context state -----------------------------------------
        self.wakewords = wakewords
        self.end_words = end_words or []
        self.awake_timeout = awake_timeout
        self.context_timeout = context_timeout
        self.barge_in_keywords = barge_in_keywords or self.DEFAULT_BARGE_IN

        self._awake: Dict[str, bool] = {}
        self._last_activity: Dict[str, float] = {}
        self._pending_end: Dict[str, bool] = {}

        # --- Transaction management ---------------------------------------
        self._active_txn: Dict[str, str] = {}
        self._interrupted_txn: set = set()

        # --- Context tracker (shared with LLM service) --------------------
        # The LLM service updates timestamps; we query them here.
        # If an external LLM was injected, it should carry its own tracker.

        # --- Callbacks (set by adapter or user) ---------------------------
        self.handle_response: Callable = self._default_handle
        self.stop_response: Callable = self._default_stop

        # --- Hooks --------------------------------------------------------
        self._on_before_llm: Callable = self._default_before_llm
        self._on_before_tts: Callable = self._default_before_tts
        self._on_finish: Callable = self._default_finish
        self._on_performance: Callable = self._default_performance

    # ==================================================================
    # Decorators for user customization
    # ==================================================================

    def on_before_llm(self, func):
        self._on_before_llm = func
        return func

    def on_before_tts(self, func):
        self._on_before_tts = func
        return func

    def on_finish(self, func):
        self._on_finish = func
        return func

    def on_performance(self, func):
        """Register a callback to receive per-invocation performance metrics.

        The callback receives ``(request: STSRequest, metrics: dict)`` where
        *metrics* contains timing breakdowns in milliseconds:

        - ``vad_to_stt_ms`` — VAD silence → STT complete
        - ``stt_to_llm_first_ms`` — STT done → LLM first text token
        - ``llm_first_to_tts_first_ms`` — LLM first token → first TTS audio
        - ``vad_to_tts_first_ms`` — **total**: VAD silence → first TTS audio
        - ``text`` — recognized / input text
        """
        self._on_performance = func
        return func

    async def _default_before_llm(self, req: STSRequest):
        pass

    async def _default_before_tts(self, req: STSRequest):
        pass

    async def _default_finish(self, req: STSRequest, resp: STSResponse):
        sid = req.session_id
        if self._pending_end.pop(sid, False):
            self._awake[sid] = False
            logger.info(f"Session {sid} awake ended by end word")

    async def _default_performance(self, req: STSRequest, metrics: dict):
        """Default handler: log a one-line performance summary."""
        parts = [
            f"VAD→STT {metrics['vad_to_stt_ms']}ms",
            f"STT→LLM {metrics.get('stt_to_llm_first_ms', '?')}ms",
            f"LLM→TTS {metrics.get('llm_first_to_tts_first_ms', '?')}ms",
            f"TOTAL {metrics.get('vad_to_tts_first_ms', '?')}ms",
        ]
        logger.info(f"[PERF] \"{metrics.get('text', '')[:30]}\" | {' | '.join(parts)}")

    # ==================================================================
    # Awake state logic
    # ==================================================================

    def _check_awake(self, request: STSRequest) -> Tuple[bool, bool]:
        """Returns ``(is_awake, just_woke_up)``."""
        sid = request.session_id
        now = time()

        # Already awake?
        if self._awake.get(sid, False):
            if now - self._last_activity.get(sid, 0) < self.awake_timeout:
                self._last_activity[sid] = now
                return True, False
            # Timed out
            self._awake[sid] = False
            logger.info(f"Session {sid} awake timed out")

        # No wake words → always awake
        if not self.wakewords:
            self._awake[sid] = True
            self._last_activity[sid] = now
            return True, False

        # Check for wake word in text
        if request.text:
            for ww in self.wakewords:
                if ww in request.text:
                    self._awake[sid] = True
                    self._last_activity[sid] = now
                    logger.info(f"Session {sid} awakened by '{ww}'")
                    return True, True

        return False, False

    def _has_end_word(self, text: str) -> bool:
        return any(ew in text for ew in self.end_words)

    def _is_barge_in(self, text: str) -> bool:
        """Check if *text* starts with an explicit interruption keyword."""
        if not text or not self.barge_in_keywords:
            return True  # No keywords → always allow
        cleaned = text.strip().lstrip("，,。！？.!?～~…")
        for kw in self.barge_in_keywords:
            if cleaned.startswith(kw):
                return True
        return False

    # ==================================================================
    # Main pipeline
    # ==================================================================

    async def invoke(self, request: STSRequest) -> AsyncGenerator[STSResponse, None]:
        """Run the full STT→LLM→TTS pipeline for one speech segment."""
        try:
            txn_id = str(uuid4())
            t_vad = time()  # T0: VAD silence confirmed → invoke starts

            # ---- 1. STT --------------------------------------------------
            if request.text:
                recognized = request.text
                if self.debug:
                    logger.info(f"Using text: {recognized}")
            elif request.audio_data:
                recognized = await self.stt.transcribe(request.audio_data)
                if not recognized:
                    if self.debug:
                        logger.info("No speech recognized")
                    return
                if self.debug:
                    logger.info(f"STT: '{recognized}'")
            else:
                recognized = ""

            request.text = recognized
            t_stt = time()  # T1: STT complete

            # ---- 2. Context expiry check ---------------------------------
            ctx_id = request.context_id
            if ctx_id:
                age = self.llm.context_tracker.get_age(ctx_id)
                if age > self.context_timeout:
                    logger.info(f"Context {ctx_id} expired ({age:.0f}s), creating new")
                    request.context_id = None

            # ---- 3. Awake check ------------------------------------------
            is_awake, just_woke = self._check_awake(request)

            # End word (skip if just woke up to avoid immediate exit)
            if not just_woke and self._has_end_word(recognized):
                self._pending_end[request.session_id] = True
                logger.info(f"End word in session {request.session_id}")

            if not is_awake:
                if self.debug:
                    logger.info(f"Not awake, skipping")
                return

            # ---- 4. Context id -------------------------------------------
            if not request.context_id:
                request.context_id = str(uuid4())
                logger.info(f"New context: {request.context_id}")

            # ---- 5. Barge-in control -------------------------------------
            prev = self._active_txn.get(request.session_id)
            if prev and prev != txn_id:
                if not self._is_barge_in(recognized):
                    logger.info(f"No barge-in keyword in '{recognized}', keeping txn {prev}")
                    return
                self._interrupted_txn.add(prev)
                logger.info(f"Barge-in by '{recognized}', interrupting {prev}")

            if self.debug:
                logger.info(f"Start txn {txn_id}: '{request.text}'")
            self._active_txn[request.session_id] = txn_id

            # ---- 6. Stop previous playback -------------------------------
            await self.stop_response(request.session_id, request.context_id)

            yield STSResponse(
                type="start",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"request_text": request.text},
            )

            # ---- 7. LLM stream -------------------------------------------
            await self._on_before_llm(request)
            llm_stream = self.llm.chat_stream(
                request.context_id, request.user_id, request.text, request.files,
                request.system_prompt_params,
            )

            # ---- 8. TTS (inlined with LLM stream) ------------------------
            t_llm_first = None  # T2: first LLM text token
            t_tts_first = None  # T3: first TTS audio

            async def synthesize():
                nonlocal t_llm_first, t_tts_first
                voice_text = ""
                language = None
                async for chunk in llm_stream:
                    if not self._is_active(request.session_id, txn_id):
                        if self.debug:
                            logger.info(f"LLM stream broken: new txn {self._active_txn.get(request.session_id)}")
                        break

                    # Skip tool calls (nanobot handles them server-side)
                    if chunk.tool_call:
                        yield None, chunk
                        continue

                    if t_llm_first is None:
                        t_llm_first = time()  # T2: first LLM text chunk

                    if chunk.voice_text:
                        voice_text += chunk.voice_text
                        if not language:
                            await self._on_before_tts(request)

                    audio = await self.tts.synthesize(
                        text=chunk.voice_text,
                        language=language,
                    )
                    if t_tts_first is None and audio:
                        t_tts_first = time()  # T3: first TTS audio generated

                    yield audio, chunk
                return

            response_text = ""
            first_chunk = True
            t_first_queued = None  # T4: first audio enqueued for playback
            async for audio, llm_chunk in synthesize():
                if not self._is_active(request.session_id, txn_id):
                    break

                if llm_chunk.tool_call:
                    yield STSResponse(
                        type="tool_call",
                        session_id=request.session_id,
                        user_id=request.user_id,
                        context_id=llm_chunk.context_id,
                        tool_call=llm_chunk.tool_call,
                    )
                    continue

                response_text += llm_chunk.text or ""

                if t_first_queued is None and audio:
                    t_first_queued = time()  # T4: first audio ready for playback

                yield STSResponse(
                    type="chunk",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=llm_chunk.context_id,
                    text=llm_chunk.text,
                    voice_text=llm_chunk.voice_text,
                    audio_data=audio,
                    metadata={"is_first_chunk": first_chunk},
                )
                first_chunk = False

            # ---- 9. Performance report ------------------------------------
            perf = {
                "text": request.text or "",
                "vad_to_stt_ms": round((t_stt - t_vad) * 1000),
                "stt_to_llm_first_ms": round((t_llm_first - t_stt) * 1000) if t_llm_first else None,
                "llm_first_to_tts_first_ms": round((t_tts_first - t_llm_first) * 1000) if (t_llm_first and t_tts_first) else None,
                "tts_first_to_first_queued_ms": round((t_first_queued - t_tts_first) * 1000) if (t_tts_first and t_first_queued) else None,
                "vad_to_tts_first_ms": round((t_tts_first - t_vad) * 1000) if t_tts_first else None,
                "vad_to_first_queued_ms": round((t_first_queued - t_vad) * 1000) if t_first_queued else None,
            }
            await self._on_performance(request, perf)

            # ---- 10. Finalize ---------------------------------------------
            self._interrupted_txn.discard(txn_id)
            if self._active_txn.get(request.session_id) == txn_id:
                del self._active_txn[request.session_id]

            final = STSResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                text=response_text,
            )
            await self._on_finish(request, final)
            yield final

        except Exception as ex:
            tb = traceback.format_exc()
            logger.error(f"Pipeline error: {ex}\n{tb}")
            yield STSResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"error": str(ex) if self.debug else "Pipeline error"},
            )

    # ==================================================================
    # Helpers
    # ==================================================================

    def _is_active(self, session_id: str, txn_id: str) -> bool:
        return self._active_txn.get(session_id) == txn_id

    async def _default_handle(self, response: STSResponse):
        logger.info(f"Response: {response.type}")

    async def _default_stop(self, session_id: str = None, context_id: str = None):
        pass

    def _setup_logging(self):
        if self.debug and not logger.hasHandlers():
            root = logging.getLogger("voice_assistant")
            root.setLevel(logging.DEBUG)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s"))
            root.addHandler(h)

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def finalize(self, context_id: str):
        await self.vad.finalize_session(context_id)

    async def shutdown(self):
        await self.llm.shutdown()
