"""Core voice assistant pipeline: VAD → STT → LLM(nanobot) → TTS → speaker.

Orchestrates the full speech-to-speech flow with wake-word detection,
barge-in interruption, and context timeout management.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import traceback
from datetime import datetime
from time import time
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from .delegate import DelegateManager
from .models import LLMResponse, STSRequest, STSResponse
from .protocol import N_DELEGATED, N_ERROR, N_NO_SPEECH, N_NO_MATCH
from .tools import (
    ASYNC_TOOL,
    DELEGATE_SPEC,
    GET_CURRENT_TIME_SPEC,
    get_current_time,
)
from .backend.vad import SpeechDetector, StandardSpeechDetector
from .backend.stt import AliyunASRSpeechRecognizer, NullSpeechRecognizer, SpeechRecognizer
from .backend.llm import DeepSeekLLMService, LLMService
from .backend.tts import (
    BaiduSpeechSynthesizer,
    CosyVoiceSpeechSynthesizer,
    EdgeTTSSpeechSynthesizer,
    NullSpeechSynthesizer,
    SpeechSynthesizer,
)
from .backend.kws import KeywordWakeDetector

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
        # --- 音频端注册表(多端仲裁: 本机声卡 / 手机等) ------------------------
        # None = 由外部(adapter)直接设置 handle_response/stop_response 的旧式用法,
        # 行为与改造前完全一致。见 transport/ 包的说明。
        transport_registry=None,
        # 无头模式: 不构造需要凭据的 STT/TTS(文本模式用; 见 stt.NullSpeechRecognizer)
        text_only: bool = False,

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
        # stt=None 显式表示"本条链路不需要语音识别"（文本模式），装一个会明确抛错的
        # 占位实现：构造不再需要 DASHSCOPE key，误接音频也不会静默失败。
        if stt is not None:
            self.stt = stt
        elif text_only:
            self.stt = NullSpeechRecognizer(debug=debug)
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
        # tts=None 同理（文本模式且不合成）：无头模式，避免去构造需要 DashScope key 的
        # CosyVoice。误走播报路径会得到一条明确的 NullTTS 错误，而不是静默无声。
        if tts is not None:
            self.tts = tts
        elif text_only:
            self.tts = NullSpeechSynthesizer(debug=debug)
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
        #: 按需唤醒的会话（远端用：手机不喊唤醒词，靠"按住说话"这个手势表达意图）
        self._wake_on_connect: Dict[str, bool] = {}

        # --- Transaction management ---------------------------------------
        self._active_txn: Dict[str, str] = {}
        #: 会话 → 该轮开始时刻（用于清理"中途异常、永远不收尾"的卡死轮次）
        self._txn_started: Dict[str, float] = {}
        self._interrupted_txn: set = set()

        # --- Context tracker (shared with LLM service) --------------------
        # The LLM service updates timestamps; we query them here.
        # If an external LLM was injected, it should carry its own tracker.

        # --- Callbacks (set by adapter or user) ---------------------------
        self.handle_response: Callable = self._default_handle
        self.stop_response: Callable = self._default_stop

        # --- 音频端注册表 --------------------------------------------------
        # 注册表在时, 下行音频按 session_id 路由到对应的端; 本机 transport 承担
        # 兜底。注册表为 None 时, handle_response/stop_response 仍是上面那两个
        # 回调(由 adapter 设置) —— 旧的单端用法一行都不用改。
        self.transport_registry = transport_registry

        # --- 无头(文本)模式标记 --------------------------------------------
        # 置位后: STT/TTS 是明确抛错的占位实现, 且不允许 pipeline 自己发声
        # (播报由对应 transport 接管)。见 transport/text.py。
        self.text_only = text_only

        # --- Hooks --------------------------------------------------------
        self._on_before_llm: Callable = self._default_before_llm
        self._on_before_tts: Callable = self._default_before_tts
        self._on_finish: Callable = self._default_finish
        self._on_performance: Callable = self._default_performance
        #: 轮次结果回调（见 on_turn_outcome）：保证每轮都有反馈，不会静默消失
        self._on_turn_outcome: Optional[Callable] = None
        #: 识别结果回调（见 on_transcript）：让客户端看见"我听成了什么"
        self._on_transcript: Optional[Callable] = None

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

    _WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    #: 模型"嘴上说去查、手上没调工具"的典型措辞。
    #: 只收**第一人称正在进行/即将进行**的动作，避免把"提议"也算进来
    #: （实测误报：「需要我帮你查什么，说一声我马上办」是提议，不是承诺）。
    PROMISE_PATTERN = re.compile(
        r"我去查|我这就去查|这就去查|我查一下|正在查|我去看看|我去问问|"
        r"稍后告诉你|马上告诉你|稍后播报|查完再告诉|查到了告诉你|"
        r"我马上去查|立刻去查"
    )
    #: 出现这些措辞说明模型在**征求意见/提议**，不算"承诺"
    OFFER_PATTERN = re.compile(r"要不要|需要我|要我帮|帮你查吗|需要的话|说一声")
    #: 命中上面模式后插入的纠正提示（不给模型留"再骗一次"的空间）
    CORRECTIVE_NOTE = (
        "（系统提醒）你刚才对用户说要查询，但**并没有实际调用任何工具**，后台也没有"
        "正在进行的任务。现在二选一：\n"
        "1) 确实需要查询 → 立刻调用 delegate_to_dsh，并且**不要再重复确认话术**"
        "（用户已经听过一遍了）；\n"
        "2) 不需要查询 → 直接回答用户，不要再声称\"正在查\"或\"稍后告知\"。"
    )

    def _should_correct_unbacked_promise(self, text: str, called_tools: List[str],
                                        session_id: str, already_corrected: bool) -> bool:
        """本轮"说了要查但没调用工具"且后台确实没任务 → 需要纠正一轮。"""
        if already_corrected or called_tools or not text:
            return False
        if not self.PROMISE_PATTERN.search(text):
            return False
        if self.OFFER_PATTERN.search(text):        # 是在提议/征求意见, 不是承诺
            return False
        if self.delegate.running(session_id):      # 后台真在跑 → 模型没说错
            return False
        return True

    def _dynamic_context(self, session_id: str = None) -> str:
        """每轮实时环境说明（当前时间 + 后台任务状态）。

        为什么必须有：
          * **当前时间**：模型不知道今天几号，会往委托任务描述里写错年份
            （实测：2026-09-19 那天它写了"2025年9月20日"）。
          * **后台任务状态**：模型不知道后台到底有没有任务，于是用户问"查到了吗"时
            只能顺着自己上一句"我去查一下"编出"还在查"（实测：用户白等 4 分钟，
            其实从未提交过任务）。
        """
        now = datetime.now()
        lines = [
            f"当前时间：{now.strftime('%Y-%m-%d %H:%M')} "
            f"{self._WEEKDAYS[now.weekday()]}。涉及日期、星期、"
            f"\"今天/明天\"时一律以此为准，不要凭记忆推断年份。"
        ]
        running = self.delegate.running(session_id)
        if running:
            waited = int(time() - running[0].started_at)
            lines.append(
                f"后台查询任务：正在进行中（已等待 {waited} 秒，任务内容："
                f"{running[0].task[:40]}）。用户问进展时如实说明；结果出来会自动播报，"
                f"不要重复提交同一任务。"
            )
        else:
            lines.append(
                "后台查询任务：当前**没有**正在进行的任务。用户问「查到了吗/还在查吗」时，"
                "如实说明现在没有在查、需要的话马上帮他查；**绝对不要**声称「还在查」。"
            )
        return "\n".join(lines)

    def attach_transport(self, registry) -> None:
        """把音频端注册表挂到 pipeline 上（多端模式；不挂就是原来的单端行为）。

        做了两件事，且**都不改变本机路径的语义**：

        1. ``handle_response`` 改走注册表 —— 按 ``STSResponse.session_id`` 路由到
           发起该轮的那个端。本机会话（``default_user``）命中 ``is_local`` 兜底，
           最终仍然落到 ``AudioDeviceAdapter.handle_response``，与改造前一致。
        2. ``_is_idle()`` 增加一个条件：**任何远端在录音时不开口**。本机侧仍然由
           VAD 回答，所以单端场景的判定逻辑与改造前完全相同。

        注册表为 None 时不调用本方法即可，旧写法（adapter 自己设回调）继续有效。
        """
        self.transport_registry = registry
        if registry is not None:
            self.handle_response = registry.handle_response
            # transport 需要注册表来**显式绑定/解绑会话**（远端接入/断开）。
            # 只靠 session_prefix 路由有个漏洞：连接断开后前缀仍然命中，
            # 于是"已消失的会话"还会被路由过去（实测发现）。显式绑定优先于前缀，
            # 解绑后让注册表回落到本机端才是正确语义。
            registry.attach_pipeline(self)

    def _is_remote_session(self, session_id: str = None) -> bool:
        """该会话是否由**远端**(手机等)承载。

        用来分流"落历史"的判定：本机路径必须保持改造前的行为（无条件写），
        只有远端会话才依据它自己汇报的"是否被打断"来决定。
        """
        if self.transport_registry is None:
            return False
        try:
            owner = self.transport_registry.resolve(session_id)
        except Exception as ex:                                   # noqa: BLE001
            logger.warning(f"解析会话归属失败: {ex}")
            return False
        return owner is not None and not owner.is_local

    def _should_remember_turn(self, session_id: str, response_text: str,
                              interrupted_playback) -> bool:
        """这一轮要不要写进对话历史。

        两条路径的语义不同，**不能混用**：

        * **本机**：改造前是「只要模型说了话就写」，保持原样。这里刻意不依赖
          ``stop_response`` 的返回值 —— ``AudioDeviceAdapter`` 是旧式回调，返回
          ``None``，若把 ``None`` 当成"没播放过"就会让助手失忆（写代码时的真实踩坑）。
        * **远端**：由远端汇报"刚才是否真的在放音被打断"来决定，避免把从未播出去的
          内容记成"说过"。旧式回调（返回 ``None``）按"那就写"处理，向后兼容。
        """
        if not response_text:
            return False
        if not self._is_remote_session(session_id):
            return True
        return interrupted_playback is None or bool(interrupted_playback)

    def _remote_recording(self, session_id: str = None) -> bool:
        """是否有**远端**(手机等)正在采集语音。没有注册表时恒为 False。"""
        if self.transport_registry is None:
            return False
        try:
            return bool(self.transport_registry.is_recording(session_id))
        except Exception as ex:                                   # noqa: BLE001
            logger.warning(f"查询远端录音状态失败: {ex}")
            return False

    def _any_remote_recording(self) -> bool:
        """**任意**远端在说话（跨会话）—— 本机用它避免对着正在说话的手机讲话。"""
        if self.transport_registry is None:
            return False
        try:
            return bool(self.transport_registry.any_remote_recording())
        except Exception:                                         # noqa: BLE001
            return False

    def _remote_playing(self, session_id: str) -> bool:
        """该远端自己是否正在放音 —— 这是远端"忙不忙"的唯一正确判据。"""
        if self.transport_registry is None:
            return False
        try:
            return bool(self.transport_registry.is_playing(session_id))
        except Exception:                                         # noqa: BLE001
            return False

    #: 一轮最多允许占用多久（秒）—— 超时即视为"卡死的轮次"。
    #: 依据：VAD 最长 10s（max_duration）+ STT/LLM/TTS 云端往返几秒，30s 有充足余量。
    TXN_STALE_AFTER = 30.0

    #: 短于这个时长（秒）的片段不送云端 STT。
    #: 实测依据：DashScope 对"太短/近静音"的音频直接返回
    #: `400 InvalidParameter: The audio is empty`（32ms 必报、1s 正常），
    #: 白跑一趟网络还必然失败；VAD 的 min_duration(0.2s) 太宽松，这里再收一道。
    MIN_STT_DURATION = 0.35

    def on_turn_outcome(self, func):
        """注册"一轮结果"回调：``async (session_id, outcome, text) -> None``。

        ``outcome`` 取值见 :mod:`voice_assistant.protocol` 的 ``N_*`` 常量。
        **每一轮都必须回调一次** —— 用户说完话后永远要得到某种回应，
        哪怕是"没听清"。实测踩到过整轮静默：语音端无从得知发生了什么，
        界面就永远停在"稍等"。
        """
        self._on_turn_outcome = func
        return func

    async def _notify_turn_outcome(self, session_id: str, outcome: str, text: str = ""):
        cb = getattr(self, "_on_turn_outcome", None)
        if cb is None:
            return
        try:
            await cb(session_id, outcome, text)
        except Exception as ex:                                   # noqa: BLE001
            logger.warning(f"上报轮次结果失败: {ex}")

    def on_transcript(self, func):
        """注册"识别结果"回调：``async (session_id, text) -> None``。

        让客户端能看见"我听成了什么"。缺了它，识别错误时用户只能看到
        "没能回答"，完全不知道是自己没说清、还是助手听错了。
        """
        self._on_transcript = func
        return func

    async def _notify_transcript(self, session_id: str, text: str):
        cb = getattr(self, "_on_transcript", None)
        if cb is None or not text:
            return
        try:
            await cb(session_id, text)
        except Exception as ex:                                   # noqa: BLE001
            logger.warning(f"上报识别结果失败: {ex}")

    def _active_txn_id(self, session_id: str = None):
        """当前进行中的轮次 id；**过期的自动清掉**。

        为什么必须有"自愈"：`_active_txn` 在 VAD 判句那一刻就被置位，而清掉它
        只发生在 `invoke` 正常收尾的路径上。一旦这一轮在中途异常（STT 401、
        网络断、TTS 抛错…），那个 txn 会**永远留在表里** → `_is_idle` 恒为假 →
        手机端"按住说话"永久按不动，界面一直显示"服务端还忙着"。
        这类"卡死的状态"没有自愈机制就会表现为不可恢复的假死（实测踩到）。
        """
        txn = self._active_txn.get(session_id)
        if txn is None:
            return None
        started = self._txn_started.get(session_id)
        if started is not None and (time() - started) > self.TXN_STALE_AFTER:
            logger.warning(
                f"清理卡死的轮次 {txn}（会话 {session_id}，已超过 "
                f"{self.TXN_STALE_AFTER:.0f}s 未收尾）"
            )
            self._active_txn.pop(session_id, None)
            self._txn_started.pop(session_id, None)
            self._kws_reset(session_id)
            return None
        return txn

    def _is_idle(self, session_id: str = None) -> bool:
        """现在适合开口吗？

        本机侧条件: 没有进行中的用户轮次、扬声器没在放音、VAD 没在录音。
        另加一条: **没有任何远端正在说话**（注册表为 None 时恒不成立）。

        ⚠️ **远端会话的"忙"要看它自己的播放状态，而不是录音**（实测踩到软锁）：
        客户端按下按钮时会先发 `start`，服务端若把"用户在说话"算进 idle，
        就会立刻回 `idle=false`；客户端据此显示"服务端还忙着"并拒绝下一次输入 ——
        **它把自己锁死了**（用它在说话这个事实，去禁止它说话）。
        所以远端只问两件事：有没有进行中的轮次、**它自己是否正在放音**
        （`is_playing`，由端汇报）。本机侧保持原样：VAD/回声抑制会吃话，
        录音时必须闭嘴。
        """
        remote = self._is_remote_session(session_id) if session_id else False
        active = self._active_txn_id(session_id)
        if remote:
            busy = bool(active) or self._remote_playing(session_id)
            return not busy

        return not (
            active
            or self.vad.should_mute()
            or self.vad.is_recording(None)
            # 本机也不该对着"正在说话的手机"讲话（跨会话的保守保护）
            or self._any_remote_recording()
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
        """把一句话说出去。

        **队列与时机**（什么时候说、忙则排队）由 pipeline 统一决定，因为它是跨端
        一致的；**怎么发声**交给对应的端 —— 本机走 TTS 合成，文本端打印，远端推给
        App。不这么做的话，文本模式（TTS 是明确抛错的占位实现）一遇到委托播报就会
        抛异常打断整轮（实测踩到过）。

        没有注册表时退回原来的行为（TTS → 播放回调），旧用法一行不用改。
        """
        logger.info(f"[announce] 播报: {text[:40]}")
        if self.transport_registry is not None:
            await self.transport_registry.emit(text, session_id)
            return

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

        # 远端会话按需唤醒（由 transport 在客户端明确表达"我要说话"后调用 request_wake）
        if self._skip_wake_check(sid):
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

    def request_wake(self, session_id: str, reason: str = ""):
        """把一个会话置为"已唤醒"（远端按需唤醒，见 :attr:`_wake_on_connect`）。

        语义：手机上"按住说话"这个**手势本身就是明确的说话意图**，唤醒词的作用
        （防止误触发）已经由它承担，不需要再喊一遍。将来手机端做本地唤醒词时，
        客户端命中唤醒词后再调用它即可 —— 服务端逻辑不变。
        """
        self._awake[session_id] = True
        self._last_activity[session_id] = time()
        logger.info(f"Session {session_id} 已按需唤醒{('（' + reason + '）' if reason else '')}")

    def _skip_wake_check(self, session_id: str) -> bool:
        """这个会话是否**绕过唤醒词检查**。

        这是唯一的判据，被两处共用（`invoke` 的 KWS 前置检查 + `_check_awake`）。
        **必须统一**：实测踩到过只改了一处 —— `_check_awake` 放行了，但 `invoke`
        开头的 KWS 前置检查仍直接看 `self._awake`，于是音频在 STT 之前就被丢掉，
        用户看到的是"按住说了没反应"（日志里 `KWS(streaming): no wake word`）。
        """
        return bool(self._wake_on_connect.get(session_id))

    #: 疑问句结尾标记：这些结尾说明用户在**提问**，不是在下"结束"指令
    QUESTION_TAILS = ("吗", "呢", "么", "嘛", "?", "？")

    def _has_end_word(self, text: str) -> bool:
        """判断是否是"结束/退下"指令。

        历史：原先用 `any(ew in text ...)` 配一份含"好了/行了/了解/收到"等日常词的
        大词表，结果「天气查好了吗？」被判成结束语 → 助手静默回到待唤醒态，用户以为
        程序退出（见 docs/混合路由规划.md Phase E1）。

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
            # 是否真的中断了正在进行的播放（由 transport 在第 7 步汇报）。初值 None =
            # "没人报过"，落历史判定按向后兼容处理；正常路径一定会被第 7 步覆盖。
            interrupted_playback = None
            # 本轮结果（见 _notify_turn_outcome）：**每一轮都必须落到某个值**，
            # 否则用户说完话后会得不到任何回应（实测踩到的"永远稍等"）。
            outcome: Optional[str] = None
            outcome_text = ""

            # ---- 1. KWS pre-check ----------------------------------------
            # 正常路径: 音频是经 feed_audio() 进来的, 流式 KWS 已经边收边判过唤醒词,
            # 这里只需要看状态 —— 没唤醒就直接丢, 不再做任何解码。
            # 只有调用方绕过了 feed_audio()(直接喂 vad.process_samples)时, 才退回
            # 整段批量判一次, 免得永远醒不过来。
            sid = request.session_id
            if request.audio_data and not self._skip_wake_check(sid):
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
                # 太短的片段不必送云端：DashScope 对近静音/过短音频直接报
                # `400 The audio is empty`（实测 32ms 必报、1s 正常），
                # 白跑网络且注定失败。直接判为"没听清"更诚实、更快。
                if request.audio_duration and request.audio_duration < self.MIN_STT_DURATION:
                    if self.debug:
                        logger.info(
                            f"片段过短（{request.audio_duration:.2f}s < "
                            f"{self.MIN_STT_DURATION}s），跳过云端 STT"
                        )
                    outcome, outcome_text = N_NO_SPEECH, "刚才太短了，没听清"
                    return
                recognized = await self.stt.transcribe(request.audio_data)
                if not recognized:
                    if self.debug:
                        logger.info("No speech recognized")
                    # ⚠️ 这里绝不能静默 return：客户端会永远停在"稍等…"，
                    # 用户以为助手死了（实测踩到）。必须给一个结果。
                    outcome, outcome_text = N_NO_SPEECH, "没听清，请再说一次"
                    return
                if self.debug:
                    logger.info(f"STT: '{recognized}'")
            else:
                recognized = ""

            request.text = recognized
            t_stt = time()  # T1: STT complete
            # 把识别结果告诉客户端（远端才需要；文本端/本机没有"听众"）
            await self._notify_transcript(request.session_id, recognized)

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
            self._txn_started[request.session_id] = time()   # 供"卡死轮次自愈"判断

            # ---- 7. Stop previous playback -------------------------------
            # 返回值 = "是否真的中断了正在进行的播放"。它有两个用途:
            #   ① 落历史(见 9.5): 只有真被打断过的那一轮才无条件写;
            #   ② 修正一处旧缺陷: 改造前每次调用都无条件写历史, 会把从未播出的
            #      内容(如开始播报前的竞态轮次)记成"说过"。见 E2 同类问题的思路。
            interrupted_playback = await self.stop_response(request.session_id, request.context_id)

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
            turn_text = ""                    # 本轮所有已排队文本(用于"承诺未落地"检测)
            dyn_note = self._dynamic_context(request.session_id)

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
                nonlocal t_llm_first, turn_text
                corrected = False
                try:
                    for _round in range(self.max_tool_rounds + 1):
                        stream = self.llm.chat_stream(
                            request.context_id, request.user_id, request.text, request.files,
                            request.system_prompt_params,
                            extra_messages=round_messages or None,
                            persist=False,
                            dynamic_context=dyn_note,
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
                                turn_text += chunk.voice_text or ""
                                await sentence_q.put(chunk)

                        if tool_hit is None:
                            # 本轮没有任何工具调用。检查"嘴上说去查、手上没调工具"——
                            # 实测踩过：模型说"我去查一下"却没发 tool_call，用户问
                            # "查到了吗"时它又编出"还在查"，白等 4 分钟。
                            if self._should_correct_unbacked_promise(
                                    turn_text, called_tools, request.session_id, corrected):
                                corrected = True
                                logger.warning("模型声称去查询但没有调用工具 → 纠正一轮")
                                round_messages.append({
                                    "role": "system", "content": self.CORRECTIVE_NOTE,
                                })
                                continue
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
                    # 注意 text/voice_text 用**本句**而不是 llm_chunk.text: 后者是
                    # "最后到达的那个 chunk", 一段话分多句时会重复/错位(实测: TTS
                    # 被连着喂了两遍整段)。播放层只读 audio_data, 所以对语音链路
                    # 无影响; 但对要拿文本做字幕/回显的端(手机/文本端)是必须的。
                    #
                    # text_only(文本模式)下 **不合成**: 没有音频后端, 而且文本端
                    # 要的是文字。这里直接把文本交给端 —— 否则会撞上 NullSpeechSynthesizer
                    # 的明确报错, 把整轮打断(实测)。
                    if self.text_only:
                        yield STSResponse(
                            type="chunk",
                            session_id=request.session_id,
                            user_id=request.user_id,
                            context_id=llm_chunk.context_id,
                            text=llm_chunk.voice_text,
                            voice_text=llm_chunk.voice_text,
                            audio_data=None,
                            metadata={"is_first_chunk": first_chunk, "source": "text"},
                        )
                        first_chunk = False
                        continue

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
                            text=llm_chunk.voice_text,
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
            # 与改造前**完全一致**的地方: 本机路径只要模型说了话就写。
            # 新增的能力: 远端会话由该端汇报"是否真的在放音被打断"来决定,
            # 避免把从未播出的内容记成"说过"。判定逻辑见 _should_remember_turn。
            if self._should_remember_turn(request.session_id, response_text,
                                          interrupted_playback):
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
                self._txn_started.pop(request.session_id, None)

            final = STSResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                text=response_text,
            )
            await self._on_finish(request, final)
            # 本轮有产出：委托类明确告知"已转后台"（客户端可据此提示等待）
            if "delegate_to_dsh" in called_tools:
                outcome, outcome_text = N_DELEGATED, "已转后台处理"
            else:
                outcome, outcome_text = "ok", (response_text or "")[:40]
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
            outcome, outcome_text = N_ERROR, (str(ex)[:80] if self.debug else "本轮出错")
            yield failed
        finally:
            # ⚠️ **异常/被取消的轮次必须立刻清理 txn**（实测踩到灾难性后果）：
            # `_active_txn` 在 VAD 判句时置位，原本只有"正常收尾"那条路径会清它。
            # 一旦这一轮中途抛错（STT 401、网络断、TTS 失败…），这个 txn 会永久
            # 留下 → `_is_idle` 恒为假 → 手机端"按住说话"再也按不动、
            # 界面永远显示"服务端还忙着" —— 只能重启进程才能恢复。
            # 放在 finally 里，异常、生成器提前关闭、任务被取消三种情况都能覆盖。
            if self._active_txn.get(request.session_id) == txn_id:
                del self._active_txn[request.session_id]
                self._txn_started.pop(request.session_id, None)
                self._interrupted_txn.discard(txn_id)
            # 每轮**必须**给客户端一个结果：正常回答 / 已转后台 / 没听清 / 出错。
            # 漏掉任何一条路径，用户都会看到界面卡在"稍等…"（实测踩到）。
            if outcome is None and request.audio_data:
                outcome, outcome_text = N_NO_MATCH, "没能回答这一轮"
            if outcome is not None:
                await self._notify_turn_outcome(request.session_id, outcome, outcome_text)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _is_active(self, session_id: str, txn_id: str) -> bool:
        # 用 _active_txn_id（带过期自愈）而不是直接查表：卡死的轮次会在这里
        # 被清掉，于是"流被打断"的分支能自然走到，而不是永远认为自己是活跃的。
        return self._active_txn_id(session_id) == txn_id

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
