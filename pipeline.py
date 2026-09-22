"""Core voice assistant pipeline: VAD → STT → LLM(nanobot) → TTS → speaker.

Orchestrates the full speech-to-speech flow with wake-word detection,
barge-in interruption, and context timeout management.
"""

import asyncio
import contextlib
import json
import logging
import os
import traceback
from time import time
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from .delegate import DelegateManager
from .models import LLMResponse, STSRequest, STSResponse
from .tools import (
    ASYNC_TOOL,
    DELEGATE_SPEC,
    GET_CURRENT_TIME_SPEC,
    get_current_time,
)
from .vad import SpeechDetector, StandardSpeechDetector
from .stt import AliyunASRSpeechRecognizer, SpeechRecognizer
from .llm import DeepSeekLLMService, LLMService
from .tts import (
    BaiduSpeechSynthesizer,
    CosyVoiceSpeechSynthesizer,
    EdgeTTSSpeechSynthesizer,
    SpeechSynthesizer,
)
from .kws import KeywordWakeDetector

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
        TTS provider: ``"cosyvoice"`` (default), ``"edge"`` (free edge-tts),
        or ``"baidu"``.
    llm_api_key:
        LLM API key (defaults to ``DEEPSEEK_API_KEY`` env var).
    llm_base_url:
        OpenAI-compatible endpoint (default ``https://api.deepseek.com``).
    llm_model:
        Model name (default ``"deepseek-flash"``).
    llm_system_prompt:
        System prompt sent with every request.
    llm_thinking:
        Enable the model's thinking mode (default False — voice turns want the
        first token fast).
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
        tts_rate: Optional[str] = None,   # edge-tts rate, e.g. "+10%" (None = default)
        tts_pitch: Optional[str] = None,  # edge-tts pitch, e.g. "+0Hz"
        tts_volume: Optional[str] = None, # edge-tts volume, e.g. "+0%"
        # LLM (direct OpenAI-compatible API)
        llm: LLMService = None,
        llm_api_key: str = None,
        llm_base_url: str = "https://api.deepseek.com",
        llm_system_prompt: str = None,
        llm_model: str = "deepseek-flash",
        llm_thinking: bool = False,
        # VAD
        vad: SpeechDetector = None,
        vad_volume_threshold: float = -40.0,
        vad_silence_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        # KWS (local keyword spotting)
        kws: "KeywordWakeDetector | None" = None,
        # Wake / context
        wakewords: List[str] = None,
        end_words: List[str] = None,
        awake_timeout: float = 60.0,
        context_timeout: float = 3600.0,
        # Barge-in
        barge_in_keywords: List[str] = None,
        # Noise filter: duration + text based (Plan D)
        noise_words: List[str] = None,
        min_meaningful_duration: float = 1.0,
        # Delegate (复杂任务委托, 走常驻中继插件)
        delegate_url: str = "http://127.0.0.1:8765",
        delegate_enabled: bool = True,
        delegate_progress_after: float = 25.0,
        delegate_progress_interval: float = 45.0,
        cancel_words: List[str] = None,
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

        # --- KWS -----------------------------------------------------------
        self.kws = kws

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

        # --- LLM (direct API) ---------------------------------------------
        if llm:
            self.llm = llm
        else:
            self.llm = DeepSeekLLMService(
                base_url=llm_base_url,
                api_key=llm_api_key,
                system_prompt=llm_system_prompt,
                model=llm_model,
                thinking=llm_thinking,
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
        elif tts_provider == "edge":
            self.tts = EdgeTTSSpeechSynthesizer(
                voice=tts_voice,
                rate=tts_rate or "+0%",
                pitch=tts_pitch or "+0Hz",
                volume=tts_volume or "+0%",
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

        # --- Noise filter ---------------------------------------------------
        self.noise_words = set(noise_words or [])
        self.min_meaningful_duration = min_meaningful_duration

        # --- Tools ----------------------------------------------------------
        # 同步工具(时间/天气/…)返回字符串 → 带着结果再问模型一次;
        # 异步工具(delegate)返回 ASYNC_TOOL → 本轮结束, 结果以后台播报送达。
        self.max_tool_rounds = 3
        # 兜底确认话术: 工具调用轮里模型经常"只发工具不吐字"(实测 content 恒为 0),
        # 于是用户要静等结果。仅当本轮一个字都没说时才补这一句, 不会和模型自己
        # 说的话重复。
        self.delegate_confirm_text = "好的，我去查一下，稍后告诉您。"
        # 取消词: 用户说"别查了"时本地直接掐掉后台任务, 不用过 LLM(零延迟又准)
        self.cancel_words = cancel_words if cancel_words is not None else [
            "别查了", "不用查了", "不查了", "停止查询", "取消查询", "别查",
        ]
        # --- 复杂任务委托(常驻中继) -----------------------------------------
        self.delegate = DelegateManager(
            base_url=delegate_url,
            announce=self.announce,
            is_idle=self._is_idle,
            remember=self.remember_from_background,
            enabled=delegate_enabled,
            progress_after=delegate_progress_after,
            progress_interval=delegate_progress_interval,
            debug=debug,
        )
        self._pending_announcements: List[Tuple[str, str]] = []
        self._announce_task: Optional[asyncio.Task] = None
        self._register_default_tools()

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
            self._kws_reset(sid)
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
    # Tools
    # ==================================================================

    def _register_default_tools(self):
        """注册内置工具(时间 + 复杂任务委托)。"""
        self.llm.add_tool("get_current_time", GET_CURRENT_TIME_SPEC, get_current_time)
        self.llm.add_tool("delegate_to_dsh", DELEGATE_SPEC, self._delegate_tool)

    async def _delegate_tool(self, task: str, metadata: dict = None):
        """复杂任务委托：交给常驻中继（qqbot profile 内的受限 dsh agent）异步执行。

        提交后**立即**返回 :data:`ASYNC_TOOL`，本轮到此结束 —— 结果由
        :class:`~voice_assistant.delegate.DelegateManager` 在后台轮询并按策略播报。
        提交失败时返回一段文本，让模型正常回话（比静默失败好）。
        """
        meta = metadata or {}
        session_id = meta.get("session_id")
        task_id = await self.delegate.submit(task, session_id, meta.get("context_id"))
        if not task_id:
            return "(委托服务当前不可用，请告诉用户稍后再试)"
        logger.info(f"[delegate] {task_id} 受理: {task[:60]}")
        return ASYNC_TOOL

    async def remember_from_background(self, context_id: str, text: str):
        """把后台播报出去的结论写进对话历史。

        否则用户听到结果后再问"刚才那个结果是多少"，前端模型会说"我还没收到结果"
        （实测确认过），体验上等于助手失忆。**进度播报不写** —— 只有最终结论值得记。
        """
        if not context_id or not text:
            return
        await self.llm.append_turn(context_id, None, text)
        logger.info(f"[delegate] 结论已写入历史: {text[:40]}")

    def _is_idle(self, session_id: str = None) -> bool:
        """现在适合开口吗？

        三个条件都必须满足: 没有进行中的用户轮次、扬声器没在放音、VAD 没在录音。
        录音中绝不能开口 —— 回声抑制会把 VAD 静音, 用户正在说的那句话会被截断。
        """
        return not (
            self._active_txn.get(session_id)
            or self.vad.should_mute()
            or self.vad.is_recording(session_id)
        )

    async def announce(self, text: str, session_id: str = None):
        """主动播报一句：空闲就直接说，忙就排队（由常驻 worker 在空闲时播）。"""
        if not text:
            return
        if not self._is_idle(session_id):
            if (session_id, text) not in self._pending_announcements:
                logger.info(f"[announce] 会话忙, 排队等空闲: {text[:30]}")
                self._pending_announcements.append((session_id, text))
            return
        await self._speak(text, session_id)

    async def _speak(self, text: str, session_id: str = None):
        """真正开口（流式合成 → 交给播放层）。"""
        logger.info(f"[announce] 播报: {text[:40]}")
        first = True
        async for audio in self.tts.synthesize_stream(text):
            await self.handle_response(STSResponse(
                type="chunk", session_id=session_id, audio_data=audio,
                metadata={"is_first_chunk": first, "sample_rate": self.tts.sample_rate,
                          "source": "announce"},
            ))
            first = False
        await self.handle_response(STSResponse(type="final", session_id=session_id))

    async def _announce_worker(self, interval: float = 0.5):
        """常驻播报调度：每 0.5s 看一次排队内容，空闲了就播。

        之前用"被挡下就 schedule 一次重试"的写法，有**重入 bug**：重试任务自己
        调 flush → 仍忙 → 再调 schedule，而此刻 _announce_retry 就是这个正在运行的
        任务（not done()）→ 直接 return → 重试链断掉，那条播报**永久丢失**。
        改成单 worker 轮询后，不存在这条自引用路径。
        """
        while True:
            await asyncio.sleep(interval)
            if not self._pending_announcements:
                continue
            sid, text = self._pending_announcements[0]
            if not self._is_idle(sid):
                continue
            self._pending_announcements.pop(0)
            try:
                await self._speak(text, sid)
            except Exception as ex:                               # noqa: BLE001
                logger.error(f"播报失败: {ex}")

    def flush_announcements(self, session_id: str = None):
        """保留的兼容入口：worker 会自己轮询，这里无需做事（但仍记一条便于排查）。"""
        if self._pending_announcements:
            logger.info(f"[announce] 待播报 {len(self._pending_announcements)} 条, 等空闲")

    # ==================================================================
    # Audio entry point
    # ==================================================================

    async def feed_audio(self, samples: bytes, session_id: str):
        """麦克风原始音频的唯一入口: 先喂流式 KWS(边收边判唤醒词), 再喂 VAD(切句)。

        流式 KWS 让唤醒判定跟着音频走 —— 等 VAD 切出句子时唤醒状态早就定了,
        对话链路上不再有 KWS 的整段解码(原来 0.7~1.5s)那段开销。
        """
        if self.kws and self.wakewords and not self._awake.get(session_id):
            self.kws.feed(session_id, samples)
            hit = self.kws.poll(session_id)
            if hit:
                self._awake[session_id] = True
                self._last_activity[session_id] = time()
                logger.info(f"Session {session_id} awakened by KWS: '{hit}'")
        await self.vad.process_samples(samples, session_id)

    def _kws_reset(self, session_id: str):
        """会话睡回去时丢掉 KWS 流式状态, 免得残留上下文干扰下一次唤醒。"""
        if self.kws is not None and hasattr(self.kws, "reset_stream"):
            self.kws.reset_stream(session_id)

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
            self._kws_reset(sid)
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

    #: 疑问句结尾标记：这些结尾说明用户在**提问**，不是在下"结束"指令
    QUESTION_TAILS = ("吗", "呢", "么", "嘛", "?", "？")

    def _has_end_word(self, text: str) -> bool:
        """判断是否是"结束/退下"指令。

        历史：原先用 `any(ew in text ...)` 配一份含"好了/行了/了解/收到"等日常词的
        大词表，结果「天气查好了吗？」被判成结束语 → 助手静默回到待唤醒态，用户以为
        程序退出（见 bt_assistant/混合路由规划.md Phase E1）。

        修法**两头一起收干净**：
          * 词表侧：run.py 的 END_WORDS 只留"本义就是停止/离开"的说法，歧义日常词
            全部剔除并注明原因（防止有人再加回来）；
          * 逻辑侧：只留一条守卫 —— **疑问句不算**。

        因此那套"按词长分层（整句/句首/句尾）"的复杂匹配已无必要：**歧义来自词表，
        不来自匹配方式**。现在就是最简单的子串匹配 + 疑问句守卫。
        """
        raw = (text or "").strip()
        if not raw or not self.end_words:
            return False
        if raw.endswith(self.QUESTION_TAILS):
            return False
        return any(ew in raw for ew in self.end_words)

    def _match_cancel_word(self, text: str) -> bool:
        """是否命中"取消后台查询"的本地词表。"""
        if not text or not self.cancel_words:
            return False
        cleaned = text.strip()
        return any(cw in cleaned for cw in self.cancel_words)

    def _is_noise(self, text: str, audio_duration: float) -> bool:
        """Check if short-duration input with noise-only text should be ignored.

        Plan D: combine duration + text.  If the audio segment is shorter than
        *min_meaningful_duration* AND the recognized text consists entirely of
        noise/filler words (e.g. "嗯", "对", "哦"), treat it as non-input.

        Short segments with meaningful content (e.g. "退下吧" at 0.51s) are
        NOT filtered — only the noise-word check triggers.
        """
        if not self.noise_words or audio_duration >= self.min_meaningful_duration:
            return False
        cleaned = text.strip().rstrip("。，！？.!?…~～, ")
        return cleaned in self.noise_words

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

            # ---- 1. KWS pre-check ----------------------------------------
            # 正常路径: 音频是经 feed_audio() 进来的, 流式 KWS 已经边收边判过唤醒词,
            # 这里只需要看状态 —— 没唤醒就直接丢, 不再做任何解码。
            # 只有调用方绕过了 feed_audio()(直接喂 vad.process_samples)时, 才退回
            # 整段批量判一次, 免得永远醒不过来。
            sid = request.session_id
            if request.audio_data and not self._awake.get(sid):
                # Only run KWS when wakewords are configured. If wakewords is
                # None or empty, the session is always awake and KWS is unnecessary.
                if self.kws and self.wakewords:
                    hit = self.kws.poll(sid)
                    if hit:
                        self._awake[sid] = True
                        self._last_activity[sid] = time()
                        logger.info(f"Session {sid} awakened by KWS: '{hit}'")
                    elif self.kws.has_stream(sid):
                        if self.debug:
                            logger.info("KWS(streaming): no wake word, skipping cloud STT")
                        return
                    else:
                        detected = await asyncio.to_thread(self.kws.detect, request.audio_data)
                        if detected:
                            self._awake[sid] = True
                            self._last_activity[sid] = time()
                            logger.info(f"Session {sid} awakened by KWS: '{detected}'")
                        else:
                            if self.debug:
                                logger.info("KWS: no wake word, skipping cloud STT")
                            return

            # ---- 2. STT --------------------------------------------------
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

            # ---- 3. Context expiry check ---------------------------------
            ctx_id = request.context_id
            if ctx_id:
                age = self.llm.context_tracker.get_age(ctx_id)
                if age > self.context_timeout:
                    logger.info(f"Context {ctx_id} expired ({age:.0f}s), creating new")
                    request.context_id = None

            # ---- 4. Awake check ------------------------------------------
            is_awake, just_woke = self._check_awake(request)

            # End word (skip if just woke up to avoid immediate exit)
            if not just_woke and self._has_end_word(recognized):
                self._pending_end[request.session_id] = True
                logger.info(f"End word in session {request.session_id}")

            if not is_awake:
                if self.debug:
                    logger.info(f"Not awake, skipping")
                return

            # ---- 4.5. Noise filter (duration + text) -----------------------
            if self._is_noise(recognized, request.audio_duration):
                logger.info(
                    f"Noise filtered: '{recognized}' ({request.audio_duration:.2f}s)"
                )
                return

            # ---- 5. 取消委托(本地判定, 零延迟) ---------------------------
            # 用户说"别查了/停止查询"时直接掐掉后台任务, 不必绕一圈 LLM。
            cancelled = self._match_cancel_word(recognized)
            if cancelled:
                task_id = self.delegate.cancel(request.session_id)
                if task_id:
                    logger.info(f"用户取消委托 {task_id}: '{recognized}'")
                    request.context_id = request.context_id or str(uuid4())
                    yield STSResponse(
                        type="start", session_id=request.session_id,
                        user_id=request.user_id, context_id=request.context_id,
                        metadata={"request_text": request.text},
                    )
                    await self.announce("好，不查了。", request.session_id)
                    final = STSResponse(
                        type="final", session_id=request.session_id,
                        user_id=request.user_id, context_id=request.context_id,
                        text="好，不查了。",
                    )
                    await self._on_finish(request, final)
                    yield final
                    return
                if self.debug:
                    logger.info("取消词命中, 但没有在跑的后台任务")

            # ---- 5.5. Context id -----------------------------------------
            if not request.context_id:
                request.context_id = str(uuid4())
                logger.info(f"New context: {request.context_id}")

            # ---- 6. Barge-in control -------------------------------------
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

            # ---- 7. Stop previous playback -------------------------------
            await self.stop_response(request.session_id, request.context_id)

            yield STSResponse(
                type="start",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"request_text": request.text},
            )

            # ---- 8. LLM stream (含工具轮次循环) ---------------------------
            # 一轮用户输入可能触发多次 LLM 请求: 模型发 tool_call → 本地执行 →
            # 带着结果再问一次。轮内消息(assistant.tool_calls + tool 结果)只在本轮
            # 请求里携带, **不落历史** —— 历史由本轮结束时 append_turn 统一写一次,
            # 保证里面永远是干净的 user/assistant 成对消息(否则历史裁剪一旦拆散
            # tool_calls 与它的结果, 下一轮请求就会因"孤儿 tool 消息"报 400)。
            await self._on_before_llm(request)

            # ---- 9. TTS (runs in parallel with the LLM stream) ------------
            # The LLM stream fills a bounded queue with sentence-sized chunks;
            # a consumer task synthesizes them in order.  Keeping synthesis out
            # of the LLM loop means we keep pulling tokens while TTS works.
            t_llm_first = None  # T2: first LLM text token
            t_llm_request = None  # T2-: HTTP 请求真正发出的时刻
            t_llm_token = None  # T2a: 首个 token(可能还没到句末, 不能合成)
            t_tts_first = None  # T3: first TTS audio
            t_first_queued = None  # T4: first audio handed to the speaker
            response_text = ""
            first_chunk = True
            tts_started = False
            sentence_q: asyncio.Queue = asyncio.Queue(maxsize=4)
            DONE = object()
            round_messages: List[Dict] = []
            called_tools: List[str] = []      # 本轮调过的工具(用于按路径分类时延)

            def _note_first_token():
                """LLM 层首个 token 到达(此时可能还没到句末标点, 合成还开不了口)。"""
                nonlocal t_llm_token
                if t_llm_token is None:
                    t_llm_token = time()

            def _note_request_start():
                """HTTP 请求真正发出 —— 与 t_stt 之间的差值就是"请求前的本地开销"
                (停上一轮播放/拆音频流、组 prompt、事件循环调度等)。"""
                nonlocal t_llm_request
                if t_llm_request is None:
                    t_llm_request = time()

            self.llm.on_first_token = _note_first_token
            self.llm.on_request_start = _note_request_start

            async def produce():
                """LLM → 切句 → 队列(不在这里合成)。工具轮次循环也在这里。"""
                nonlocal t_llm_first
                try:
                    for _round in range(self.max_tool_rounds + 1):
                        stream = self.llm.chat_stream(
                            request.context_id, request.user_id, request.text, request.files,
                            request.system_prompt_params,
                            extra_messages=round_messages or None,
                            persist=False,
                        )
                        tool_hit = None
                        round_spoke = False
                        async for chunk in stream:
                            if not self._is_active(request.session_id, txn_id):
                                if self.debug:
                                    logger.info(
                                        f"LLM stream broken: new txn "
                                        f"{self._active_txn.get(request.session_id)}"
                                    )
                                return
                            if t_llm_first is None:
                                t_llm_first = time()  # T2: first LLM text chunk
                            if chunk.tool_call:
                                tool_hit = chunk.tool_call
                                await sentence_q.put(chunk)
                                continue
                            if chunk.voice_text:
                                round_spoke = True
                                await sentence_q.put(chunk)

                        if tool_hit is None:
                            return

                        logger.info(f"Tool call: {tool_hit.name}({tool_hit.arguments})")
                        called_tools.append(tool_hit.name)
                        try:
                            result = await self.llm.execute_tool(
                                tool_hit.name, dict(tool_hit.arguments or {}),
                                metadata={"session_id": request.session_id,
                                          "context_id": request.context_id},
                            )
                        except Exception as ex:                       # noqa: BLE001
                            logger.error(f"工具 {tool_hit.name} 执行失败: {ex}")
                            result = f"(工具执行失败: {ex})"

                        if result is ASYNC_TOOL:
                            # 委托类工具: 本轮到此结束, 结果以后台播报送达。
                            if not round_spoke:
                                # 模型一个字都没说 → 补一句兜底确认话术, 否则用户
                                # 会在结果回来前一直听不到任何声音。
                                logger.info("工具轮无文本输出 → 补兜底确认话术")
                                await sentence_q.put(LLMResponse(
                                    context_id=request.context_id,
                                    text=self.delegate_confirm_text,
                                    voice_text=self.delegate_confirm_text,
                                ))
                            if self.debug:
                                logger.info("异步委托已受理 → 本轮结束, 结果后台播报")
                            return

                        round_messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_hit.id or "call_0",
                                "type": "function",
                                "function": {"name": tool_hit.name,
                                             "arguments": json.dumps(tool_hit.arguments or {},
                                                                     ensure_ascii=False)},
                            }],
                        })
                        round_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_hit.id or "call_0",
                            "content": str(result),
                        })
                    logger.warning(f"工具轮次超过上限 {self.max_tool_rounds}, 强制结束当前轮")
                finally:
                    await sentence_q.put(DONE)

            producer = asyncio.create_task(produce())
            try:
                while True:
                    llm_chunk = await sentence_q.get()
                    if llm_chunk is DONE:
                        break
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

                    if not tts_started:
                        tts_started = True
                        await self._on_before_tts(request)

                    # 流式合成: 首个音频包一到就交给扬声器, 不等整句
                    async for audio in self.tts.synthesize_stream(llm_chunk.voice_text):
                        if not self._is_active(request.session_id, txn_id):
                            break
                        if t_tts_first is None:
                            t_tts_first = time()  # T3: first TTS audio generated
                        if t_first_queued is None:
                            t_first_queued = time()  # T4: first audio ready to play

                        yield STSResponse(
                            type="chunk",
                            session_id=request.session_id,
                            user_id=request.user_id,
                            context_id=llm_chunk.context_id,
                            text=llm_chunk.text,
                            voice_text=llm_chunk.voice_text,
                            audio_data=audio,
                            metadata={
                                "is_first_chunk": first_chunk,
                                "sample_rate": self.tts.sample_rate,
                            },
                        )
                        first_chunk = False
            finally:
                # 收尾 producer: 被中断就取消; 正常结束时 await 它,
                # 好让 LLM 的异常(如 401)在这里抛出、走 invoke 的错误分支,
                # 而不是变成 "Task exception was never retrieved"。
                if not producer.done():
                    producer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer
                # 本轮音频结束由末尾的 type="final" 通知播放层(见 adapter)

            # ---- 9.5 落历史(整轮只写一次, 不含工具管道消息) ----------------
            if response_text:
                await self.llm.append_turn(request.context_id, request.text, response_text)

            # ---- 10. Performance report -----------------------------------
            # 路径分类: 复杂路径看的是"提问→**确认话术**出声"; 最终答案属异步播报, 不计入。
            if "delegate_to_dsh" in called_tools:
                path = "delegate"
            elif called_tools:
                path = "tool"
            else:
                path = "chat"
            perf = {
                "text": request.text or "",
                "path": path,
                "tools": called_tools,
                "vad_to_stt_ms": round((t_stt - t_vad) * 1000),
                "stt_to_llm_request_ms": round((t_llm_request - t_stt) * 1000) if t_llm_request else None,
                "llm_request_to_token_ms": round((t_llm_token - t_llm_request) * 1000) if (t_llm_request and t_llm_token) else None,
                "stt_to_llm_token_ms": round((t_llm_token - t_stt) * 1000) if t_llm_token else None,
                "stt_to_llm_first_ms": round((t_llm_first - t_stt) * 1000) if t_llm_first else None,
                "llm_token_to_sentence_ms": round((t_llm_first - t_llm_token) * 1000) if (t_llm_token and t_llm_first) else None,
                "llm_first_to_tts_first_ms": round((t_tts_first - t_llm_first) * 1000) if (t_llm_first and t_tts_first) else None,
                "tts_first_to_first_queued_ms": round((t_first_queued - t_tts_first) * 1000) if (t_tts_first and t_first_queued) else None,
                "vad_to_tts_first_ms": round((t_tts_first - t_vad) * 1000) if t_tts_first else None,
                "vad_to_first_queued_ms": round((t_first_queued - t_vad) * 1000) if t_first_queued else None,
            }
            await self._on_performance(request, perf)

            # ---- 11. Finalize ---------------------------------------------
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

            # 本轮结束后看看有没有攒着的主动播报(空闲才播, 否则自己排下一次)
            self.flush_announcements(request.session_id)

        except Exception as ex:
            tb = traceback.format_exc()
            logger.error(f"Pipeline error: {ex}\n{tb}")
            failed = STSResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"error": str(ex) if self.debug else "Pipeline error"},
            )
            await self._on_finish(request, failed)
            yield failed

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
        self._kws_reset(context_id)

    async def warmup(self):
        """启动预热: 提前建好 LLM 的 httpx/SSL 上下文和 TTS 连接。

        这些一次性开销(建 SSL 上下文约 1s, TTS 首次建 websocket 约 0.3~0.5s)
        如果留到第一轮对话里, 就直接变成首字延时 —— 所以启动时先跑一遍。
        """
        await self.llm.warmup()
        await self.tts.warmup()
        # 常驻播报调度(必须在事件循环里启动): 排队内容等空闲自动播出
        if self._announce_task is None or self._announce_task.done():
            self._announce_task = asyncio.create_task(self._announce_worker())

    async def shutdown(self):
        if self._announce_task is not None:
            self._announce_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._announce_task
            self._announce_task = None
        await self.delegate.close()
        await self.llm.shutdown()
        await self.tts.close()
