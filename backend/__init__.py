"""模型与设备能力层（backend）—— 助手"用什么去听/去说/去想"。

这一层只提供**能力**，不含编排：谁先谁后、什么时候开口、失败了怎么办，
都属于 ``pipeline`` / ``delegate`` / ``transport``。

| 模块 | 能力 | 外部依赖 |
|---|---|---|
| ``adapter`` | 本机声卡 I/O（麦克风采集线程 + 扬声器播放） | PyAudio |
| ``stt`` | 语音识别（DashScope qwen3-asr-flash）+ 无头占位实现 | dashscope |
| ``tts`` | 语音合成（CosyVoice 流式 / Edge / 百度）+ 无头占位实现 | dashscope / ffmpeg |
| ``llm`` | OpenAI 兼容流式 LLM 客户端（tool_calls 累加、切句、TTS 文本清洗） | httpx |
| ``vad`` | 振幅阈值 VAD（判句） | 无 |
| ``kws`` | 本地唤醒词（sherpa-onnx 流式） | sherpa-onnx / pypinyin |

**旧导入路径仍可用**（``voice_assistant.vad`` 等 → 自动指向 ``backend.vad``），
这是为了不让内部相对导入与历史脚本因为这次搬运而断掉；新代码请直接用
``voice_assistant.backend.xxx``。
"""

import sys as _sys

from . import adapter, kws, llm, stt, tts, vad          # noqa: F401

#: 旧名 → 新模块（为兼容保留；新代码不要再用旧名）
_LEGACY_ALIASES = {
    "adapter": adapter, "vad": vad, "kws": kws,
    "stt": stt, "tts": tts, "llm": llm,
}


def install_legacy_aliases(package: str = __name__.rsplit(".", 1)[0]) -> None:
    """把 ``voice_assistant.vad`` 之类的旧路径指向 ``backend.vad``。

    为什么需要：``tests/``、``transport/``、``backend/`` 内部存在大量
    ``from ..vad import ...`` / ``from voice_assistant.backend.llm import ...`` 写法，
    这次只是"搬运"，不该逼着所有调用方同时改导入 —— 那正是最容易踩的坑。
    """
    for old, mod in _LEGACY_ALIASES.items():
        key = f"{package}.{old}"
        if key not in _sys.modules:
            _sys.modules[key] = mod


__all__ = ["adapter", "kws", "llm", "stt", "tts", "vad", "install_legacy_aliases"]
