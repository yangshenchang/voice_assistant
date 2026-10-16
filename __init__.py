from .pipeline import VoiceAssistant

# 模型/设备能力层（backend/）统一出口。
# 这里额外安装"旧路径别名"（voice_assistant.vad → backend.vad），
# 让历史脚本与内部相对导入在目录整理后继续可用。
from . import backend as backend                                   # noqa: F401

backend.install_legacy_aliases(__name__)

__all__ = ["VoiceAssistant", "backend"]
