"""本机音频端：麦克风 + 扬声器（把现有 AudioDeviceAdapter 包成 Transport）。

**阶段 0 的硬要求：行为与改造前逐字一致。**

因此这里只做"委托"，不做任何逻辑改动：

* ``start`` → ``AudioDeviceAdapter.start_listening``（麦克风读线程 + 队列那套照旧）
* ``handle_response`` → 原样转发给 adapter（它负责入播放队列）
* ``stop_response`` → 转发，并把"是否**真的**中断了播放"这一事实回传

### 为什么 stop_response 要算"是否真的中断了"

改造前 pipeline 每轮开头都会调一次 ``adapter.stop_response()``；adapter 内部有
优化（没流、没在放、队列空时直接返回，见 ``adapter/__init__.py`` 的注释），
但**调用方拿不到这个事实**，于是历史写入是无条件的。

现在接口把它变成返回值，语义更准：**只有真被打断过的那一轮，才需要把"我播报过
什么"记进历史**。单端场景下最终写入的内容与改造前相同（`invoke` 末尾那次
``append_turn`` 是主路径），只是不再产生重复条目。
"""

from typing import TYPE_CHECKING, Optional

from ..models import STSResponse
from . import AudioTransport

if TYPE_CHECKING:                                     # 只为类型标注，避免导入期副作用
    from ..backend.adapter import AudioDeviceAdapter


class LocalTransport(AudioTransport):
    """本机声卡端。

    Parameters
    ----------
    adapter:
        :class:`~voice_assistant.adapter.AudioDeviceAdapter` 实例（会构造 PyAudio）。
    """

    type = "local"
    session_prefix = ""          # 空前缀 = 不参与前缀匹配，只靠 is_local 兜底
    is_local = True

    def __init__(self, adapter: "AudioDeviceAdapter"):
        self._adapter = adapter

    @property
    def adapter(self) -> "AudioDeviceAdapter":
        """底层设备适配器（给需要直接碰设备的代码用，如 ``adapter.close()``）。"""
        return self._adapter

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self, session_id: str, user_id: str = None):
        await self._adapter.start_listening(session_id, user_id)

    async def close(self):
        self._adapter.close()

    # ------------------------------------------------------------------
    # 上行 / 下行（全部原样委托）
    # ------------------------------------------------------------------

    async def feed_audio(self, chunk: bytes, session_id: str):
        # 走 pipeline 的统一音频入口：顺路喂流式 KWS，再进 VAD 切句。
        # （改造前 adapter 直接调 sts.feed_audio，这里保持不变。）
        await self._adapter.sts.feed_audio(chunk, session_id)

    async def handle_response(self, response):
        await self._adapter.handle_response(response)

    async def emit(self, text: str, session_id: str = None):
        """本机发声：走 TTS 流式合成 → 播放层（与改造前的 ``_speak`` 完全一致）。

        这里刻意保留原来那段"边合成边推音频"的逻辑，只是搬到了端这一侧 ——
        因为"怎么发声"是端相关的，而"什么时候发声"仍由 pipeline 的播报队列决定。
        """
        sts = self._adapter.sts
        first = True
        async for audio in sts.tts.synthesize_stream(text):
            await sts.handle_response(STSResponse(
                type="chunk", session_id=session_id, audio_data=audio,
                metadata={"is_first_chunk": first, "sample_rate": sts.tts.sample_rate,
                          "source": "announce"},
            ))
            first = False
        await sts.handle_response(STSResponse(type="final", session_id=session_id))

    async def stop_response(self, session_id: str = None, context_id: str = None) -> bool:
        adapter = self._adapter
        # 与 adapter.stop_response 的内部快路径判断保持一致（否则"是否中断"
        # 的结论会和它实际做的事不一致）。用 getattr 取值，避免因属性改名而炸。
        playing_stream = getattr(adapter, "_play_stream", None)
        playing = bool(getattr(adapter, "_playing", False))
        q = getattr(adapter, "_response_queue", None)
        queued = (q is not None) and (not q.empty())
        interrupted = bool(playing_stream is not None or playing or queued)
        await adapter.stop_response(session_id, context_id)
        return interrupted

    # ------------------------------------------------------------------
    # 仲裁
    # ------------------------------------------------------------------

    def is_recording(self, session_id: Optional[str] = None) -> bool:
        """本机是否在录音 —— 由 VAD 回答。

        说明：``pipeline._is_idle`` 已经直接问了 ``vad.is_recording()``，
        这里再暴露一份是为了让**任何**持有 transport 的代码都能统一问"谁在说话"，
        不必知道本机侧的实现细节。
        """
        try:
            return bool(self._adapter.sts.vad.is_recording(session_id))
        except Exception:                                         # noqa: BLE001
            return False

    def is_playing(self, session_id: Optional[str] = None) -> bool:
        """本机扬声器是否正在放音（本机"忙不忙"的判据之一）。"""
        adapter = self._adapter
        return bool(getattr(adapter, "_playing", False)
                    or getattr(adapter, "_play_stream", None) is not None)
