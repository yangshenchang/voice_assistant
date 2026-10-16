"""远端音频端（手机 / 平板 / 网页）—— **阶段 0 只定接口，不实现通道**。

## 为什么现在只放骨架

阶段 0 的目标是"把接缝抽出来、本机行为不变"。远端真正能跑起来还缺两样：
**协议定稿**（阶段 2）和**接入入口**（阶段 5）。所以这里先把**形状**定下来，
让 pipeline / 注册表 / 自检脚本都能按最终形态编写，阶段 1 填进 WebSocket 实现
时不需要回头改接口。

## 这个类要回答的三个问题（阶段 1 实现要点）

1. **音频从哪来** → ``feed_audio``：把 WSS 收到的 PCM 块交出去。
   手机端保持 **16k / 16bit / mono / 512 帧（32ms）**，与本机完全同格式，
   这样 VAD 判句粒度与既有基线的时延特性可以直接沿用（零重采样、零解码）。

2. **音频到哪去** → ``handle_response``：把 TTS 的 PCM 原样推回该会话的连接。
   注意 pipeline 产出的 ``STSResponse`` 已带 ``session_id``，按它路由即可；
   ``metadata["sample_rate"]`` 必须一并带上——手机侧要拿它开播放器。

3. **谁在说话** → ``is_recording``：本机侧由 VAD 回答，**远端只能由远端报告**。
   推荐实现：记录"最后一次收到上行音频的时刻"，在
   ``RECORDING_WINDOW``（约 0.4~0.6s，与 VAD 的静音判句阈值同量级）内算作在录音。
   这是主动播报不被用户话打断的关键——不实现它，手机用户会在说话中途被播报插嘴。

## 一个必须继承的行为约定

``stop_response`` **必须返回"是否真的中断了播放"**。手机端"正在播放 TTS"这个
事实只有远端自己知道，返回值决定这一轮要不要写进历史（见
:class:`~voice_assistant.transport.local.LocalTransport` 的说明）。
"""

import time
from typing import Optional

from . import AudioTransport

#: 判定"远端还在说话"的时间窗（秒）。与 VAD 的静音判句阈值同量级。
RECORDING_WINDOW = 0.5


class RemoteTransport(AudioTransport):
    """远端音频端的公共基类。

    子类需要实现 :meth:`_send_response` / :meth:`_stop_playback` 两个"真正干活"
    的钩子，以及 :meth:`feed_audio` 的上游接线；本类负责把
    "最后一次上行音频时刻 → 是否在录音"这条最容易写错的逻辑固化下来。
    """

    type = "remote"
    #: 远端会话必须带前缀，避免与本机会话（``default_user``）混淆。
    session_prefix = "phone:"
    is_local = False

    def __init__(self, *, recording_window: float = RECORDING_WINDOW):
        self.recording_window = recording_window
        #: session_id → 最后一次收到上行音频的时刻（单调时钟）
        self._last_uplink: dict = {}
        #: session_id → 远端自己报告的"正在播放"
        self._playing: dict = {}

    # ------------------------------------------------------------------
    # 子类必须实现的钩子
    # ------------------------------------------------------------------

    async def _send_response(self, response):
        """把 pipeline 的响应（含音频 chunk）送给该会话的远端。"""
        raise NotImplementedError

    async def _stop_playback(self, session_id: str = None) -> bool:
        """通知远端停止播放，返回**是否真的中断了**正在进行的播放。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 通用实现
    # ------------------------------------------------------------------

    def note_uplink(self, session_id: str):
        """收到一个上行音频块时调用（由具体连接的接收循环负责）。

        从音频**到达服务端**的时刻起算，而不是等 VAD 判句 —— 判句会晚
        0.5s 左右，那时用户早已说完，用它当"在说话"的信号会让主动播报
        有机会插进用户话里。
        """
        self._last_uplink[session_id] = time.monotonic()

    def note_playback(self, session_id: str, playing: bool):
        """远端报告自己的播放状态（如收到客户端的 ``playback/state`` 事件）。"""
        self._playing[session_id] = bool(playing)

    def is_recording(self, session_id: Optional[str] = None) -> bool:
        now = time.monotonic()
        if session_id is not None:
            last = self._last_uplink.get(session_id)
            return last is not None and (now - last) < self.recording_window
        return any(
            (now - t) < self.recording_window for t in self._last_uplink.values()
        )

    def forget(self, session_id: str):
        """断开某会话时清理状态（否则它的"最后上行时刻"会一直留着）。"""
        self._last_uplink.pop(session_id, None)
        self._playing.pop(session_id, None)

    # ------------------------------------------------------------------
    # AudioTransport 接口
    # ------------------------------------------------------------------

    async def handle_response(self, response):
        session_id = getattr(response, "session_id", None)
        if session_id is not None and getattr(response, "type", None) == "chunk":
            # 服务端开始往这个会话送音频 → 远端随即进入播放态
            self.note_playback(session_id, True)
        elif session_id is not None and getattr(response, "type", None) == "final":
            self.note_playback(session_id, False)
        await self._send_response(response)

    async def stop_response(self, session_id: str = None, context_id: str = None) -> bool:
        interrupted = bool(self._playing.get(session_id)) if session_id else any(self._playing.values())
        if await self._stop_playback(session_id):
            interrupted = True
        if session_id is not None:
            self._playing[session_id] = False
        else:
            self._playing.clear()
        return interrupted
