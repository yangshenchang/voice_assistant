"""文本端：不起声卡、不走 STT，直接用文本跟助手对话。

## 为什么需要它

改造前想验一轮对话，**必须对着麦克风说话**。这带来三个实际代价：

1. 没有 API key / 没有声卡的场合（如持续集成、远程排查）**完全无法验证**；
2. 每改一处逻辑（prompt、切句、工具轮次、委托播报），都要"说一遍再听一遍"；
3. 手机端协议还在定稿，却没有任何不依赖音频的入口可以先跑通链路。

文本端把这三点一起解决：它复用**同一条** ``pipeline.invoke`` 与**同一套**
transport 装配，只把"音频从哪来"换成了 stdin。

```
本机:  麦克风 → VAD → STT ─┐
文本:  stdin ──────────────┴→ pipeline.invoke → LLM/工具/委托 → 输出
```

## 它同时是一张"架构体检表"

文本端是 ``AudioTransport`` 的第三个实现（本机 / 远端 / 文本）。它能跑通，
就说明 pipeline 确实只依赖接口、不再依赖声卡 —— 这正是阶段 0 想验证的性质。

## 与语音端的差异（有意为之，不是遗漏）

* **不写入任何音频缓冲**：``stop_response`` 恒返回 ``False``（没有播放可中断），
  于是"是否中断过播放"不会污染落历史判定（见 ``pipeline._should_remember_turn``）。
* **唤醒状态预置为已唤醒**：文本端不经过 KWS，若沿用"必须先喊唤醒词"的规则，
  第一句话就会被丢掉（``_check_awake`` 会返回未唤醒）。因此这里直接把会话置为
  已唤醒，并按 ``awake_timeout`` 正常续期。
* **不产生音频指标**：没有 VAD/STT，所以 ``vad_to_*`` 那几个字段为空；
  文本端改用 ``text_to_llm_first_ms`` 等以**提问时刻**为起点的指标（见 ``ask``）。
"""

import asyncio
import logging
import sys
from time import time
from typing import Callable, List, Optional

from ..models import STSRequest, STSResponse
from . import AudioTransport

logger = logging.getLogger(__name__)


class TextTransport(AudioTransport):
    """stdin/stdout 文本端。

    Parameters
    ----------
    sts:
        :class:`~voice_assistant.pipeline.VoiceAssistant` 实例。
    session_id:
        会话 id，默认 ``text_user``。
    user_id:
        用户 id，默认与 *session_id* 相同。
    synthesize:
        是否仍调用 TTS 合成。默认 True —— 保持与语音链路**同一段代码路径**
        （``_speak`` / 播报队列都会被真正执行），只是音频交给本类后不播放。
        置 False 可让"不配 TTS key 也能跑文本"。
    echo:
        是否把助手的回复打到 stdout（默认 True）。
    input_fn:
        读取一行输入的钩子（默认 ``input()``）。测试时注入脚本化输入，
        免得自测脚本必须喂 stdin。
    """

    type = "text"
    session_prefix = "text:"
    is_local = True

    def __init__(
        self,
        sts,
        *,
        session_id: str = "text_user",
        user_id: str = None,
        synthesize: bool = True,
        echo: bool = True,
        input_fn: Optional[Callable[[str], str]] = None,
    ):
        self.sts = sts
        self.session_id = session_id
        self.user_id = user_id or session_id
        self.synthesize = synthesize
        self.echo = echo
        self._input_fn = input_fn or input
        self._ensure_session()

        # —— 文本端收集到的"这一轮到底说了什么" ——
        # 为什么要自己收集：``_speak``（主动播报）只往播放层推音频，模型文本
        # 不会回传给调用方。文本端既然要打印，就得自己记账。
        self._reply = ""
        self._announcements: List[str] = []
        self._perf: Optional[dict] = None
        #: 文本端独有的延迟打点（语音端用 VAD 判句时刻当起点，文本端没有）
        self._t_prompt: Optional[float] = None
        self._t_first_token: Optional[float] = None
        self._t_first_sentence: Optional[float] = None

    # ------------------------------------------------------------------
    # 会话准备
    # ------------------------------------------------------------------

    def _ensure_session(self):
        """VAD 里预建会话数据：user_id + 直接置为已唤醒。

        文本端不经过 KWS，若不置为已唤醒，第一句话会被 ``_check_awake`` 丢掉
        （表现像"助手没反应"）。按 ``awake_timeout`` 正常续期，超时后由
        ``_check_awake`` 自动处理（文本端每轮都会刷新活动时间）。
        """
        setter = getattr(self.sts.vad, "set_session_data", None)
        if setter is None:
            return
        setter(self.session_id, "user_id", self.user_id, create=True)
        self.sts._awake[self.session_id] = True
        self.sts._last_activity[self.session_id] = time()

    # ------------------------------------------------------------------
    # AudioTransport 接口
    # ------------------------------------------------------------------

    async def start(self, session_id: str = None, user_id: str = None):
        """文本端没有"开始监听"这一步（交互循环由 :meth:`run` 驱动）。"""
        return

    async def close(self):
        return

    async def feed_audio(self, chunk: bytes, session_id: str):
        """文本端不接收音频 —— 这是接口形状上的占位。

        显式实现（而不是留给基类抛 NotImplementedError）是为了让"误把音频喂给
        文本端"这类接线错误**只留一条日志**，而不是在对话中途炸掉整轮。
        """
        logger.debug("TextTransport 收到音频块(%d 字节)，已忽略", len(chunk or b""))

    async def handle_response(self, response: STSResponse):
        """接收本轮的所有输出：模型文本、音频块（若有）、主动播报。

        回复文本**一定要在这里累加**，不能只在"打印那一段"里累加 —— 有两条路径
        的 ``final`` 是不带文本的：

        * **异步委托**：本轮以兜底确认话术结束（``sentence_q`` 里塞了一条
          ``LLMResponse``），最终 ``final.text`` 来自 ``response_text``，**包含**
          确认话术（实测如此）；
        * **主动播报** ``_speak``：它自己发的 ``final`` 没有任何文本。

        所以判据是"有没有文本"而不是"是不是 final"，并且 chunk 必须无条件累加，
        否则 ``echo=False`` 与 ``ask()`` 会拿到空回复。
        """
        rtype = getattr(response, "type", None)
        if rtype == "chunk":
            if response.text:
                self._reply += response.text
                # 第一个"句子级"文本块到达 = 可以送去合成/朗读的时刻
                if self._t_first_sentence is None and self._t_prompt is not None:
                    self._t_first_sentence = time()
            if (response.audio_data and self.synthesize and self.echo
                    and not getattr(self, "_audio_noted", False)):
                self._audio_noted = True        # 只报一次，别刷屏
                rate = (response.metadata or {}).get("sample_rate", "?")
                logger.debug("TTS 首个音频包已就绪 (%sHz)，文本端不播放", rate)
        elif rtype == "final":
            # 只有**带文本**的 final 才是"本轮说完了"。不带文本的 final 是主动播报的
            # 结束信号（``emit`` 会补一个），此时什么都不该做 ——
            # 实测踩坑：把打印放在这个 if 之外，播报结束时会用**残留的上一轮 _reply**
            # 再打一遍，看起来就像"助手把确认话术说了两遍"。
            if response.text:
                self._reply = response.text
                if self.echo:
                    print(f"🤖 {self._reply.strip()}")
                    sys.stdout.flush()
                    self._ask_printed = True

    async def stop_response(self, session_id: str = None, context_id: str = None) -> bool:
        """文本端没有播放，永远返回 False（"没有中断过播放"）。"""
        return False

    def is_recording(self, session_id: str = None) -> bool:
        """文本端没有"正在说话"的持续状态：一轮输入即一次提问，问完就说完了。"""
        return False

    # ------------------------------------------------------------------
    # 主动播报（委托进度 / 结果）
    # ------------------------------------------------------------------

    async def emit(self, text: str, session_id: str = None):
        """文本端发声 = 打印出来。

        必须由端来发声（而不是让 pipeline 调自己的 ``_speak``）：文本模式若没开
        合成，pipeline 装的是 ``NullSpeechSynthesizer``（**明确抛错**的占位，见
        tts/__init__.py）—— 委托结果播报一到就抛异常打断整轮。这在真机上是必现路径。

        注意"**何时**发声"仍由 pipeline 的播报队列与 ``_is_idle`` 仲裁决定，
        这里只接管"怎么发声"。
        """
        if self.echo:
            print(f"🤖 {text.strip()}")
            sys.stdout.flush()
        # 补一个 final，让播放层/计时状态机与语音路径对齐（虽然这里没有音频）
        await self.handle_response(STSResponse(
            type="final", session_id=session_id or self.session_id,
        ))

    # ------------------------------------------------------------------
    # 性能打点（文本端专用）
    # ------------------------------------------------------------------

    def _on_llm_first_token(self):
        if self._t_first_token is None:
            self._t_first_token = time()

    async def _on_perf(self, request, metrics: dict):
        """把文本端的时间轴并进 pipeline 的指标里，再由原回调落盘。

        关键点：**必须复用同一个 dict**（pipeline 传下来的那个），因为
        ``PerfRecorder.record`` 是按 key 读的；新造一个 dict 会丢掉 pipeline
        已有的字段，或者需要重复布线。
        """
        m = metrics
        if self._t_prompt is not None and self._t_first_token is not None:
            m["text_to_llm_first_ms"] = round((self._t_first_token - self._t_prompt) * 1000)
        if self._t_prompt is not None and self._t_first_sentence is not None:
            m["text_to_first_sentence_ms"] = round((self._t_first_sentence - self._t_prompt) * 1000)
        self._perf = dict(m)
        prev = getattr(self, "_prev_perf", None)
        if prev is not None:
            await prev(request, m)

    # ------------------------------------------------------------------
    # 对外：问一句
    # ------------------------------------------------------------------

    async def ask(self, text: str) -> str:
        """把 *text* 当作一轮用户输入跑完整条链路，返回助手这一轮说的话。

        走的是 ``pipeline.invoke`` —— 与语音**完全相同**的那条路
        （工具轮次、委托、承诺纠正、落历史、时延打点全都在内）。
        """
        text = (text or "").strip()
        if not text:
            return ""
        self._reply = ""
        self._t_prompt = time()
        self._t_first_token = None
        self._t_first_sentence = None
        self._audio_noted = False
        self._ask_printed = False
        self._ensure_session()

        req = STSRequest(
            session_id=self.session_id,
            user_id=self.user_id,
            context_id=self.sts.vad.get_session_data(self.session_id, "context_id"),
            text=text,
            audio_data=None,
            audio_duration=0.0,
        )
        error = None
        async for resp in self.sts.invoke(req):
            # ⚠️ 必须自己把响应交给 transport —— `invoke` 是**生成器**：pipeline 只
            # 负责 yield，真正"投递给音频端"的那一步在内部消费者 `_on_speech` 里
            # （pipeline.py 的 `async for resp in self.invoke(...)` → handle_response）。
            # 本机路径是由 `adapter.start_listening` → `sts._on_speech` 派生任务来消费；
            # 文本端既然自己驱动 invoke，就得自己走这一步，否则响应全部落空
            # （实测踩到：TTS 都合成了、回复却是空串）。
            await self.handle_response(resp)
            if resp.type == "start":
                # 与语音路径一致：把 context_id 记到会话上，多轮才有上下文
                self.sts.vad.set_session_data(
                    self.session_id, "context_id", resp.context_id
                )
            if resp.type == "final" and (resp.metadata or {}).get("error"):
                error = (resp.metadata or {})["error"]

        # 打印**在 ask 里**补齐（handle_response 也会打，但那条路径可能因为 final
        # 无文本而没打出来），用 `_ask_printed` 去重，保证一轮只出现一行。
        if self.echo and not getattr(self, "_ask_printed", False):
            print(f"🤖 {(self._reply or '(本轮无文字输出)').strip()}")
            sys.stdout.flush()
        self._ask_printed = False
        if error:
            logger.error("本轮出错: %s", error)
        return self._reply

    async def wait_for_background(self, timeout: float = 60.0,
                                  poll: float = 0.5) -> int:
        """等后台委托任务跑完，并把期间产生的播报打印出来。

        为什么需要：``ask()`` 只覆盖"本轮"。委托是**异步**的 —— 提交后本轮立即
        以确认话术结束，结果与进度要好几十秒后才由 :meth:`emit` 送达。一次性问答
        （``run.py --ask``）如果问完就退出，这些内容会全部丢掉（实测：中继明明跑完
        并推了 QQ，本地却什么都没看到）。

        Returns
        -------
        int
            等到结束的任务数（0 = 本来就没有后台任务）。
        """
        delegate = getattr(self.sts, "delegate", None)
        if delegate is None or not hasattr(delegate, "running"):
            return 0
        pending = delegate.running(self.session_id)
        if not pending:
            return 0

        ids = {t.id for t in pending}
        print(f"⏳ 后台任务 {len(ids)} 个在跑，最多等 {timeout:.0f}s（Ctrl+C 可提前退出）…")
        deadline = time() + timeout
        try:
            while time() < deadline:
                await asyncio.sleep(poll)
                left = [t for t in delegate.running(self.session_id) if t.id in ids]
                if not left:
                    break
            # 给常驻 worker 一点时间自己播（它每 0.5s 看一次队列），
            # 之后只兜底**仍留在队列里**的那些，避免和 worker 重复播报。
            for _ in range(6):
                await asyncio.sleep(0.5)
                if not self.sts._pending_announcements:
                    break
            await self._drain_queue()
        except asyncio.CancelledError:
            raise
        return len(ids)

    async def _drain_queue(self) -> int:
        """兜底播出**还留在队列里**的播报，返回播出条数。

        竞态防御（实测踩到过）：常驻 worker 与本方法都想播同一条。判据是
        "**仍在队列里**" —— worker 播之前会先把它 ``pop(0)`` 出队，所以已播过的
        不会再次出现在队列中；反过来说，只要它还在队列里，就说明还没人播过。
        """
        q = self.sts._pending_announcements
        spoken = 0
        while q:
            sid, text = q.pop(0)
            await self.emit(text, sid)
            spoken += 1
        return spoken

    # ------------------------------------------------------------------
    # 交互循环
    # ------------------------------------------------------------------

    async def run(self, on_exit: Callable = None, wait_background: float = 0.0):
        """读 stdin，逐行提问，直到 EOF / ``exit`` / ``quit``。

        用 ``asyncio.to_thread(input)`` 读输入：同步阻塞的 ``input()`` 会卡住
        事件循环 —— 这正是本机 adapter 上踩过的同类坑（阻塞读让 SSE 增量被
        晚处理，凭空多出 ~600ms，见 `低时延规划.md` 4.3）。

        *wait_background* > 0 时，每次提问后会等后台委托任务跑完并播出进度/结果；
        否则退出时异步播报会全丢。
        """
        print(f"⌨️  文本模式（会话 {self.session_id}）｜输入 exit/quit 或 Ctrl+D/Ctrl+C 退出")
        if not self.synthesize:
            print("   （TTS 已关闭：只输出文字，不调用语音合成）")
        try:
            while True:
                try:
                    line = await asyncio.to_thread(self._input_fn, "\n你: ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                line = (line or "").strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit", "退出"):
                    break
                await self.ask(line)
                if wait_background > 0:
                    # 不等的话，委托结果/进度会随着进程退出一起消失
                    await self.wait_for_background(timeout=wait_background)
        finally:
            if on_exit is not None:
                on_exit()
