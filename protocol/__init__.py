"""语音链路协议层 —— **纯数据编解码 + 常量，不含任何 I/O**。

## 为什么单独成层

同一份协议要被**两个客户端**实现：现在的网页验证端（PWA）和将来的原生 Android
App。如果协议逻辑写在各自的 I/O 代码里，两份实现必然漂移（字段名、版本号、
分块大小、错误码各写各的）。所以这里只放"纯函数 + 常量"：

```
protocol/__init__.py   常量 + 编解码（本文件）
transport/websocket.py 服务端 I/O（用本层编解码）
web/*.js               网页端 I/O（按本层约定实现）
android/...            将来的原生端（同一约定）
```

**判据**：本模块不 import ``asyncio`` / ``websockets`` / 任何服务端依赖，
因此可以脱离网络单测（见 ``verify/f2_protocol.py``）。

## 传输约定

一条 WSS 长连，**复用三件事**：上行音频、下行 TTS 音频、服务端主动播报。

| 方向 | 帧类型 | 用途 |
|---|---|---|
| C→S | **二进制** | 上行麦克风 PCM（带 4 字节头，见下） |
| C→S | 文本(JSON) | 控制：握手、开始/结束说话、心跳、取消 |
| S→C | **二进制** | 下行 TTS PCM（带 4 字节头） |
| S→C | 文本(JSON) | 事件：就绪、识别结果、状态机、错误 |

### 二进制帧头（4 字节）

```
byte 0      : 类型 0x01=上行音频(手机→服务端)  0x02=下行音频(服务端→手机)
byte 1      : 保留(0)
bytes 2-3   : 小端 uint16 序号(取模回绕) —— 用于丢帧检测与时延排查
```

**为什么音频不压缩、不做 JSON/base64**：整条链路统一
16k / 16bit / mono 原始 PCM —— 与本机 `AudioDeviceAdapter` 的采集格式完全一致，
因此 VAD 判句粒度与既有基线（`低时延规划.md` 4.2）的时延特性可以直接沿用，
零重采样、零解码。带宽约 256 kbps，可接受。base64 会平白多 33% 体积与编解码开销。

**为什么分块 512 帧（32ms）**：与本机 ``input_chunk_size=512`` 一致。
"""

import json
import struct
from typing import Any, Dict, Optional, Tuple

#: 协议版本。**客户端与服务端必须一致** —— 手机 App 升级后与旧服务端错配时，
#: 靠它在握手阶段就明确报错，而不是等到音频行为诡异才发现。
PROTOCOL_VERSION = 1

#: 音频格式（全链路统一；改动它等于改协议，必须升版本号）
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2          # 16bit
CHANNELS = 1
FRAME_SAMPLES = 512       # 每块帧数 = 32ms
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH

# 二进制帧类型
BIN_UPLINK = 0x01         # 手机 → 服务端：麦克风 PCM
BIN_DOWNLINK = 0x02       # 服务端 → 手机：TTS PCM
BIN_HEADER = struct.Struct("<BBH")   # type, reserved, seq
BIN_HEADER_SIZE = BIN_HEADER.size    # 4

# 客户端 → 服务端
C_HELLO = "hello"         # {type, version, token, sampleRate, frameSamples}
C_START = "start"         # 开始一轮说话（用户按下）
C_END = "end"             # 结束一轮说话（用户松开 / VAD 判句）
C_PING = "ping"           # {t}: 延迟探测
C_CANCEL = "cancel"       # 取消当前轮（打断）

# 服务端 → 客户端
S_READY = "ready"         # {sessionId, sampleRate, frameSamples, protocol, speaking}
S_STT = "stt"             # {text}: 识别结果（给用户看"我说了什么"）
S_TEXT = "text"           # {text, sentence?}: 助手这一轮说的话
S_ANNOUNCE = "announce"   # {text}: 主动播报（委托进度/结果）
S_STATE = "state"         # {idle, speaking, recording}: 状态机（客户端据此决定能否推流）
S_PONG = "pong"           # {t}: 回显客户端时间戳
S_ERROR = "error"         # {code, message}
S_BYE = "bye"             # {reason}: 服务端主动断开（如鉴权失败/重复连接）
S_NOTICE = "notice"       # {code, message}: 本轮结果提示（"没听清"这类，非错误）
S_TURN = "turn"           # {phase: start|end, outcome?}: 轮次边界（客户端做进度反馈）

# 提示码（客户端按 code 决定文案，服务端也给 message 兜底）
N_NO_SPEECH = "no_speech"       # 没识别到内容（说得太短/太轻/环境噪音）
N_NO_MATCH = "no_answer"        # 识别到了但没产生回复
N_DELEGATED = "delegated"       # 已转后台处理，稍后播报
N_ERROR = "turn_error"          # 本轮出错

# 错误码（客户端应按 code 而不是 message 做判断）
E_BAD_VERSION = "bad_version"
E_UNAUTHORIZED = "unauthorized"
E_BAD_MESSAGE = "bad_message"
E_DUPLICATE = "duplicate_session"
E_INTERNAL = "internal"


# ---------------------------------------------------------------------------
# JSON 信封：客户端 → 服务端
# ---------------------------------------------------------------------------

def encode(msg_type: str, **fields) -> str:
    """编码一条客户端→服务端的 JSON 文本帧。"""
    payload: Dict[str, Any] = {"type": msg_type}
    payload.update({k: v for k, v in fields.items() if v is not None})
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """解析一条 JSON 文本帧 → ``(type, fields)``。

    解析失败返回 ``(None, {})``，由调用方按 :data:`E_BAD_MESSAGE` 回错；
    这里**不抛异常** —— 坏帧不该打断整条连接（它可能只是客户端 bug 或扫描器流量）。
    """
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None, {}
    if not isinstance(obj, dict):
        return None, {}
    msg_type = obj.pop("type", None)
    if not isinstance(msg_type, str) or not msg_type:
        return None, {}
    return msg_type, obj


# ---------------------------------------------------------------------------
# 事件构造：服务端 → 客户端（统一走这里，避免字段名各处手写成两套）
# ---------------------------------------------------------------------------

def ready(session_id: str, speaking: bool = False) -> str:
    return json.dumps({
        "type": S_READY,
        "protocol": PROTOCOL_VERSION,
        "sessionId": session_id,
        "sampleRate": SAMPLE_RATE,
        "frameSamples": FRAME_SAMPLES,
        "speaking": speaking,
    }, ensure_ascii=False, separators=(",", ":"))


def state(protocol_idle: bool, speaking: bool, recording: bool) -> str:
    """状态机：客户端只在 ``idle=True`` 且 ``speaking=False`` 时接受新推流/播报。

    ``idle`` 的含义与 ``pipeline._is_idle`` 一致（没有进行中的轮次、没在放音、
    没在录音），是服务端对"现在能不能开口"的唯一权威回答。
    """
    return json.dumps({
        "type": S_STATE, "idle": bool(protocol_idle),
        "speaking": bool(speaking), "recording": bool(recording),
    }, ensure_ascii=False, separators=(",", ":"))


def text_message(msg_type: str, text: str, **extra) -> str:
    payload = {"type": msg_type, "text": text}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def error(code: str, message: str = "") -> str:
    return json.dumps({"type": S_ERROR, "code": code, "message": message},
                      ensure_ascii=False, separators=(",", ":"))


def bye(reason: str) -> str:
    return json.dumps({"type": S_BYE, "reason": reason},
                      ensure_ascii=False, separators=(",", ":"))


def pong(client_ts) -> str:
    return json.dumps({"type": S_PONG, "t": client_ts},
                      ensure_ascii=False, separators=(",", ":"))


def notice(code: str, message: str = "") -> str:
    """本轮结果提示（非错误）。

    存在意义：**绝不能让一轮输入无声无息地消失**。实测踩到：用户说了一句话，
    服务端识别为空直接 `return`，客户端什么都没收到 → 界面永远停在"稍等…"，
    用户以为助手死了。哪怕只是"没听清，请再说一次"，也必须说出来。
    """
    return json.dumps({"type": S_NOTICE, "code": code, "message": message},
                      ensure_ascii=False, separators=(",", ":"))


def turn(phase: str, outcome: str = None, text: str = None) -> str:
    """轮次边界事件：``phase`` ∈ {"start", "end"}。

    客户端用它做进度反馈（"正在识别…/正在回答…"），而不是只能看到"稍等"。
    """
    payload = {"type": S_TURN, "phase": phase}
    if outcome:
        payload["outcome"] = outcome
    if text:
        payload["text"] = text
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 二进制音频帧
# ---------------------------------------------------------------------------

def pack_audio(pcm: bytes, seq: int, downlink: bool = True) -> bytes:
    """给 PCM 加上 4 字节头。*seq* 取模回绕（uint16）。"""
    kind = BIN_DOWNLINK if downlink else BIN_UPLINK
    return BIN_HEADER.pack(kind, 0, seq & 0xFFFF) + pcm


def unpack_audio(frame: bytes) -> Tuple[Optional[int], int, bytes]:
    """拆开二进制帧 → ``(类型, 序号, PCM)``；帧太短返回 ``(None, 0, b"")``。"""
    if len(frame) < BIN_HEADER_SIZE:
        return None, 0, b""
    kind, _reserved, seq = BIN_HEADER.unpack_from(frame, 0)
    return kind, seq, frame[BIN_HEADER_SIZE:]


def is_expected_audio_size(pcm: bytes) -> bool:
    """上行音频块大小是否合规（允许最后一块不满，但必须是偶数个 16bit 采样）。"""
    return bool(pcm) and len(pcm) % SAMPLE_WIDTH == 0


def frames_to_ms(n_frames: int) -> float:
    """帧数 → 毫秒（时延换算用，避免各处重复算）。"""
    return n_frames * 1000.0 / SAMPLE_RATE
