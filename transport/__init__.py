"""音频传输层 —— 把"音频从哪来、到哪去"从 pipeline 里彻底剥出来。

## 为什么要有这一层

改造前，``adapter.AudioDeviceAdapter`` 和 pipeline 是**硬连**的：它在构造函数里
直接改写 ``sts.handle_response`` / ``sts.stop_response``（见 ``adapter/__init__.py``）。
于是"音频必须来自本机声卡"这件事被焊死在编排逻辑里，想加第二个音频端
（手机 / 平板 / 网页）就无处下手。

这一层的作用就是让 pipeline 只认一个**接口**，不认具体设备：

```
pipeline ──┬─ LocalTransport    （本机麦克风 + 扬声器，行为与改造前逐字一致）
           └─ RemoteTransport   （手机等远端：WSS 上行音频 / 下行 TTS / 主动播报）
```

## 三个必须由这层负责的语义

1. **按会话路由**：pipeline 每轮产出的音频必须送到"发起这轮的那个端"。
   本机说的送到扬声器，手机说的走 WSS —— 不能串台。
2. **仲裁（谁在说话）**：多端同时在线时，主动播报（委托结果、进度）要判定
   "现在适合开口吗"。本机侧靠 VAD 判，远端侧只能由**远端自己**报告，
   所以 :meth:`AudioTransport.is_recording` 是接口的一部分。
3. **落历史的归属**：只有**真正被打断/停止过播放**的那一端才需要记录"我播报过
   什么"。改造前本机每轮都无条件记一次，多端场景下那会污染远端会话的历史。

## 与本机行为的关系（重要）

阶段 0 的目标是**行为零变化**：``LocalTransport`` 只是把现有
``AudioDeviceAdapter`` 包了一层并原样委托，pipeline 编排、VAD、KWS、STT、
LLM、TTS、委托播报的逻辑一行未改。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

__all__ = ["AudioTransport", "TransportRegistry"]


class AudioTransport(ABC):
    """一个音频端（本机声卡、手机、网页…）的抽象。"""

    #: 传输类型标识（日志/自检用）
    type: str = "abstract"
    #: 会话前缀。注册表按前缀把 pipeline 的会话路由到对应端；
    #: 远端应给会话加前缀（如 ``phone:xxx``），避免与本机会话 `default_user` 混淆。
    session_prefix: str = ""
    #: 是否是本机端（注册表把它当**兜底**：没有别的端认领会话时就用它）
    is_local: bool = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self, session_id: str, user_id: str = None):
        """建立会话并开始收音频（阻塞到结束/被取消）。"""
        raise NotImplementedError

    async def close(self):
        """释放设备/连接。"""

    async def warmup(self):
        """可选：启动预热（建连等一次性开销别落在第一轮里）。"""

    # ------------------------------------------------------------------
    # 上行：音频 → pipeline
    # ------------------------------------------------------------------

    async def feed_audio(self, chunk: bytes, session_id: str):
        """把一个音频块交给说话人识别流程（VAD 切句 → STT → LLM → TTS）。

        由 transport 主动调用；pipeline 不关心它来自声卡还是网络。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 下行：pipeline → 音频
    # ------------------------------------------------------------------

    async def handle_response(self, response):
        """消费一个 :class:`~voice_assistant.models.STSResponse`（chunk 带 PCM 音频）。"""
        raise NotImplementedError

    async def stop_response(self, session_id: str = None, context_id: str = None) -> bool:
        """请求停止该会话的播放并丢弃排队音频。

        Returns
        -------
        bool
            是否**真的中断了正在进行的播放**。

        返回值是必需的，不是装饰：调用方（pipeline）据此决定要不要把
        "本轮播报的内容"写进对话历史。本机路径多数轮次其实没在放音，
        无条件写历史会把从来没播过的话记成"说过"。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 仲裁：现在适合开口吗
    # ------------------------------------------------------------------

    def is_recording(self, session_id: str = None) -> bool:
        """该端当前是否正在采集用户语音（有未判句的输入）。

        主动播报必须满足"没有任何端在录音" —— 否则用户正在说的那句话会被
        回声抑制静音或被打断。本机侧由 VAD 回答，远端侧只能由远端自己报。
        """
        return False

    def is_active(self, session_id: str = None) -> bool:
        """该端此刻是否真的承载着 *session_id*（或任意会话，当 *session_id* 为 None）。

        **必须实现**，这关系到播报能不能开口：只看 :attr:`session_prefix` 会把
        "已经断开的会话"也算作活跃 —— 那样仲裁会一直认为"远端有人在说话"，
        于是主动播报（委托进度/结果）**永远不播**。这类静默失效极难排查。
        """
        return True

    def is_playing(self, session_id: str = None) -> bool:
        """该端此刻是否正在**向用户播声音**（TTS 音频正在送）。

        这个事实只有端自己知道：服务端只知道"我发出去了"，不知道用户那边缓冲
        里还剩多少。远端用它回答"我这边正忙着说"，从而决定要不要接新的一轮。
        """
        return False

    # ------------------------------------------------------------------
    # 发声：谁去"说"这句话
    # ------------------------------------------------------------------

    async def emit(self, text: str, session_id: str = None):
        """把 *text* 说出来（本机 = 走 TTS 合成；文本端 = 打印；远端 = 推给 App）。

        pipeline 的播报**队列与时机**（忙则排队、空闲才说、优先级）是**跨端统一**
        的，所以留在 pipeline；而"怎么发声"是端相关的，必须由端实现。

        为什么不能直接让 pipeline 调自己的 ``_speak``：那样文本模式会走到
        ``NullSpeechSynthesizer``（明确抛错的占位）→ 委托播报一到就打断整轮。
        实测踩到过。
        """
        raise NotImplementedError


class TransportRegistry:
    """多端注册表：按会话路由 + 提供仲裁所需的全局状态。

    路由规则（顺序很重要）：

    1. **显式绑定**：``bind(session_id, transport)`` 建立的一对一映射；
    2. **前缀匹配**：会话名以某个端的 :attr:`AudioTransport.session_prefix` 开头；
    3. **本机兜底**：以上都没命中 → 交给 ``is_local`` 的那个端（``default_user``
       这类本机会话永远走它，因此**单端场景行为与改造前完全一致**）。
    """

    def __init__(self, default: Optional[AudioTransport] = None):
        self._transports: List[AudioTransport] = []
        self._default: Optional[AudioTransport] = None
        self._bindings: Dict[str, AudioTransport] = {}
        self._sts = None
        if default is not None:
            self.register(default, default=True)

    def attach_pipeline(self, sts):
        """pipeline 反向引用：transport 需要通过注册表**显式绑定/解绑会话**。

        为什么不让每个 transport 自己存一份注册表引用：注册表是"谁拥有哪个会话"
        的唯一权威，多存几份迟早不一致。统一从这里取。

        为什么远端必须显式绑定：**只靠 session_prefix 路由有漏洞** —— 连接断开后
        前缀仍然命中（比如 ``phone:abc`` 还是以 ``phone:`` 开头），于是"已消失的
        会话"还会被路由过去（实测发现）。显式绑定优先于前缀；解绑后注册表回落到
        本机端，才是正确语义。
        """
        self._sts = sts

    @property
    def pipeline(self):
        return self._sts

    # ------------------------------------------------------------------
    # 注册与绑定
    # ------------------------------------------------------------------

    def register(self, transport: AudioTransport, default: bool = False) -> AudioTransport:
        if transport not in self._transports:
            self._transports.append(transport)
        if default or self._default is None:
            self._default = transport
        return transport

    def bind(self, session_id: str, transport: AudioTransport):
        """把某个会话显式绑到某个端（远端建连时用）。"""
        self._bindings[session_id] = transport

    def unbind(self, session_id: str):
        self._bindings.pop(session_id, None)

    @property
    def transports(self) -> List[AudioTransport]:
        return list(self._transports)

    @property
    def remote_transports(self) -> List[AudioTransport]:
        """所有非本机端（仲裁时只需问它们，本机侧由 VAD 回答）。"""
        return [t for t in self._transports if not t.is_local]

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def resolve(self, session_id: str = None) -> Optional[AudioTransport]:
        if session_id and session_id in self._bindings:
            return self._bindings[session_id]
        if session_id:
            for t in self._transports:
                if t.session_prefix and session_id.startswith(t.session_prefix):
                    return t
        return self._default

    def owner(self, session_id: str = None) -> Optional[AudioTransport]:
        """**真的**拥有该会话的端（要求显式绑定或"该端自称活跃"）。

        与 :meth:`resolve` 的区别：``resolve`` 是发送路由（可以有前缀兜底），
        ``owner`` 用于**判断/仲裁**（必须精确）。两者混用会造成"已断开的会话
        仍然算活跃"，进而让主动播报永远不开口。
        """
        if session_id is None:
            return self._default
        if session_id in self._bindings:
            return self._bindings[session_id]
        for t in self._transports:
            if t.session_prefix and session_id.startswith(t.session_prefix):
                return t if t.is_active(session_id) else None
        return self._default

    # ------------------------------------------------------------------
    # 下行 / 仲裁 / 生命周期（pipeline 只跟注册表打交道）
    # ------------------------------------------------------------------

    async def handle_response(self, response):
        t = self.resolve(getattr(response, "session_id", None))
        if t is not None:
            await t.handle_response(response)

    async def stop_response(self, session_id: str = None, context_id: str = None) -> bool:
        """停掉**该会话对应的端**，并返回是否真的中断了正在进行的播放。

        为什么必须按会话路由而不是广播：本机在放音时手机发起了一轮，
        如果广播停止会把本机的话掐掉；反之亦然。
        """
        t = self.resolve(session_id)
        if t is None:
            return False
        return await t.stop_response(session_id, context_id)

    async def emit(self, text: str, session_id: str = None):
        """把播报交给该会话对应的端发声（没有注册表/没有端时静默放弃）。"""
        if not text:
            return
        t = self.resolve(session_id)
        if t is None:
            return
        await t.emit(text, session_id)

    def is_playing(self, session_id: str = None) -> bool:
        """该会话对应的端是否正在播声音（用于"这一轮能不能收新输入"）。"""
        t = self.owner(session_id) if session_id else None
        if t is not None and not t.is_local:
            try:
                return bool(t.is_playing(session_id))
            except Exception:                                     # noqa: BLE001
                return False
        return False

    def any_remote_recording(self) -> bool:
        """**任意**远端是否在说话（跨会话）。

        本机会话需要它：本机不该对着"正在说话的手机"讲话。
        """
        for t in self.remote_transports:
            try:
                if t.is_active() and t.is_recording(None):
                    return True
            except Exception:                                     # noqa: BLE001
                continue
        return False

    def is_recording(self, session_id: str = None) -> bool:
        """是否有远端正在采集语音（本机侧由 pipeline 自己问 VAD）。

        语义：**宁可不播，也不打断用户**。

        * 会话已绑定/可前缀命中**且该端真的活跃** → 只问那一个端（精确）；
        * 否则问所有远端"有没有谁在说话"—— 因为"本机问一句话、手机同时在说"
          这种情况仍不应插嘴。

        注意用 :meth:`owner` 而不是 :meth:`resolve`：后者对前缀做无条件的兜底命中，
        会把"已断开的会话"当成活跃远端，导致播报永远不开口（见 :meth:`owner` 说明）。
        """
        owner = self.owner(session_id) if session_id else None
        if owner is not None and not owner.is_local:
            try:
                return bool(owner.is_recording(session_id))
            except Exception:                                     # noqa: BLE001
                return False
        for t in self.remote_transports:
            try:
                if t.is_active() and t.is_recording(None):
                    return True
            except Exception:                                     # noqa: BLE001
                continue
        return False

    async def close(self):
        for t in self._transports:
            try:
                await t.close()
            except Exception:                                     # noqa: BLE001
                pass
        self._bindings.clear()
