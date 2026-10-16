"""服务端远端音频端：手机 / 网页通过一条 WSS 长连接进来。

## 职责边界（与 ``protocol`` 的关系）

* ``protocol/`` —— 编解码与常量（纯逻辑，可离线测）
* **本文件** —— 网络 I/O：握手鉴权、上行入队、下行发送队列、状态机、静态页

## 三条必须做对的事（都来自本机路径踩过的坑）

1. **上行音频不能在 WS 读协程里同步处理**。本机 adapter 当年就是"同步阻塞读跑在
   事件循环上"，实测让 LLM 段凭空多 600ms（`低时延规划.md` 4.3）。这里网络读本身
   是异步的，但 *送进 pipeline* 这一步仍可能被下游拖住 —— 所以用队列 + 丢最老，
   保证读协程只做"入队"这一件轻事。

2. **下行要有独立发送队列，满则丢最老**。远端网速慢时，若一个 chunk 一个 chunk
   地 ``await send``，读协程与 TTS 流水线都会被拖住。丢最老 = 宁可让用户听到
   "跳过一小段"，也不让整条链路积压到秒级延迟。

3. **"在说话"要由客户端报告**。本机侧 VAD 能回答"用户是否在说话"，远端不行；
   而 ``pipeline._is_idle`` 依赖它来决定"能不能播报" —— 不实现这个，手机用户
   会在说话中途被委托播报插嘴。

## 回声抑制

浏览器端在播放期间自行静音麦克风（``cancel_echo`` 的对等物）。服务端不做、
也做不了这件事（服务端拿不到手机扬声器/麦克风的声学关系），所以协议里把
"是否在放音"作为状态明确告知客户端。
"""

import asyncio
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import unquote, urlsplit

import websockets
from websockets.asyncio.server import serve as ws_serve
from websockets.http11 import Response
from websockets.datastructures import Headers

from ..models import STSRequest, STSResponse
from ..protocol import (
    BIN_UPLINK,
    E_BAD_MESSAGE,
    E_BAD_VERSION,
    E_DUPLICATE,
    E_UNAUTHORIZED,
    FRAME_BYTES,
    PROTOCOL_VERSION,
    SAMPLE_RATE,
    C_CANCEL,
    C_END,
    C_HELLO,
    C_PING,
    C_START,
    S_ANNOUNCE,
    S_NOTICE,
    S_STT,
    S_TEXT,
    S_TURN,
    N_DELEGATED,
    N_ERROR,
    N_NO_MATCH,
    N_NO_SPEECH,
    bye,
    decode,
    error,
    notice,
    pack_audio,
    pong,
    ready,
    state,
    text_message,
    turn,
    unpack_audio,
)
from .remote import RemoteTransport

logger = logging.getLogger(__name__)

#: 上行队列长度（块）。32ms/块 → 满队列约 4s 音频；满了丢最老，避免越堆越旧。
UPLINK_QUEUE_SIZE = 128
#: 下行队列长度（块）。约 1.3s 音频；网络慢时宁可跳帧也不积压。
DOWNLINK_QUEUE_SIZE = 40
#: 播报单独一条队列：委托结果不该被 TTS 长回复挤掉
ANNOUNCE_QUEUE_SIZE = 12
#: 空闲多久算"用户不在说话"（客户端没显式 end 时的兜底）
SPEAK_IDLE_TIMEOUT = 8.0

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".ico": "image/x-icon",
}


class RemotePerf:
    """远端时延打点：把"网络往返"和"服务端链路"分开记，否则慢在哪看不出来。"""

    def __init__(self, path: Optional[str] = None, debug: bool = False):
        self.path = Path(path) if path else None
        self.debug = debug
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.samples: List[dict] = []

    def record(self, **fields):
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        row.update(fields)
        self.samples.append(row)
        if self.debug:
            logger.info("[远端] " + " ".join(f"{k}={v}" for k, v in fields.items()))
        if self.path:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as ex:                                  # noqa: BLE001
                logger.warning("写远端基线失败: %s", ex)


class Connection:
    """一条远端连接的全部状态。"""

    def __init__(self, ws, session_id: str, transport: "RemoteTransport"):
        self.ws = ws
        self.session_id = session_id
        self.transport = transport
        self.uplink: asyncio.Queue = asyncio.Queue(maxsize=UPLINK_QUEUE_SIZE)
        self.downlink: asyncio.Queue = asyncio.Queue(maxsize=DOWNLINK_QUEUE_SIZE)
        self.announce: asyncio.Queue = asyncio.Queue(maxsize=ANNOUNCE_QUEUE_SIZE)
        self.speaking = False          # 用户在说话（start..end）
        self.playing = False           # 服务端正在给它送音频
        self.uplink_seq = 0
        self.downlink_seq = 0
        self.last_uplink_at = 0.0
        self.first_uplink_at: Optional[float] = None   # 本轮"开口时刻"（时延起点）
        self.sent_first_audio_at: Optional[float] = None
        self.announce_playing = False
        self.closed = False

    def offer(self, q: asyncio.Queue, item) -> bool:
        """非阻塞入队；满则丢最老的一块并返回 False（表示发生了丢弃）。"""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass
            return False
        try:
            q.put_nowait(item)
            return True
        except asyncio.QueueFull:
            return False


class RemoteTransport(RemoteTransport):
    """WSS 服务端 transport：管理所有远端连接，并把自己注册进 ``TransportRegistry``。

    命名说明：这里继承的是 ``transport/remote.py`` 里的基类（同名不同模块），
    基类负责"录音窗/播放态/中断回报"这些与传输无关的语义。
    """

    type = "websocket"
    session_prefix = "phone:"
    is_local = False

    def __init__(
        self,
        sts,
        *,
        host: str = "0.0.0.0",
        port: int = 8801,
        token: str = "",
        serve_web: bool = True,
        web_dir: Optional[str] = None,
        perf: Optional[RemotePerf] = None,
        tls: bool = False,
        tls_dir: Optional[str] = None,
        tls_hosts: Optional[List[str]] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.sts = sts
        self.host = host
        self.port = port
        self.token = token or ""
        self.serve_web = serve_web
        self.web_dir = Path(web_dir) if web_dir else (Path(__file__).resolve().parent.parent / "web")
        self.perf = perf
        self.tls = tls
        self.tls_dir = Path(tls_dir) if tls_dir else None
        self.tls_hosts = list(tls_hosts or [])
        self.tls_fingerprint = ""
        self.debug = debug
        self._conns: Dict[str, Connection] = {}
        self._server = None
        self._sink_tasks: Dict[str, asyncio.Task] = {}
        self._pump_tasks: Dict[str, asyncio.Task] = {}
        # 接管"本轮结果"上报：**每轮都必须给客户端一个明确交代**，
        # 否则识别为空/出错时用户什么都收不到，界面永远停在"稍等…"（实测踩到）。
        # 放在这里而不是 run.py，是为了让测试与生产走同一条装配路径。
        if hasattr(sts, "on_turn_outcome"):
            sts.on_turn_outcome(self._on_turn_outcome)
        # 识别结果也要透传：否则识别错时用户只看到"没能回答"，
        # 完全不知道是自己没说清还是助手听错了。
        if hasattr(sts, "on_transcript"):
            sts.on_transcript(self._on_transcript)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start_server(self):
        """启动监听。

        注意：中继插件那里有过"控制面失败只告警"的教训，但这里**必须显式失败** ——
        没有远端服务就等于手机连不上，属核心功能，静默降级只会让人以为是手机的问题。
        """
        kwargs = {}
        if self.tls:
            from .tls import build_ssl_context, ensure_self_signed, print_fingerprint

            cert, key = ensure_self_signed(self.tls_dir, self.tls_hosts, debug=self.debug)
            kwargs["ssl"] = build_ssl_context(cert, key)
            self.tls_fingerprint = print_fingerprint(cert)
            logger.info("TLS 已启用（自签证书 %s，指纹 %s）", cert, self.tls_fingerprint)

        self._server = await ws_serve(
            self._handler,
            self.host,
            self.port,
            process_request=self._process_http if self.serve_web else None,
            ping_interval=20,
            ping_timeout=20,
            max_size=1 << 20,          # 上行单帧上限 1MB（远大于 32ms 音频）
            compression=None,          # 音频不压缩：省 CPU，也避免压缩缓冲引入延迟
            **kwargs,
        )
        scheme = "wss" if self.tls else "ws"
        logger.info("远端服务监听 %s://%s:%d", scheme, self.host, self.port)

    async def close(self):
        for task in list(self._sink_tasks.values()) + list(self._pump_tasks.values()):
            task.cancel()
        self._sink_tasks.clear()
        self._pump_tasks.clear()
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:                                     # noqa: BLE001
                pass
            self._server = None
        for conn in list(self._conns.values()):
            try:
                await conn.ws.close()
            except Exception:                                     # noqa: BLE001
                pass
        self._conns.clear()

    @property
    def url_hint(self) -> str:
        return f"ws://<本机IP>:{self.port}"

    # ------------------------------------------------------------------
    # HTTP：静态测试页（与 WS 同端口，手机只需记一个地址）
    # ------------------------------------------------------------------

    async def _process_http(self, connection, request):
        """非 WebSocket 的普通 HTTP 请求 → 提供网页测试端。

        ⚠️ **必须放行 WebSocket 升级请求**：``process_request`` 在**握手之前**被调用，
        如果在这里无脑按静态文件处理，``/ws`` 会因"找不到文件"返回 404，客户端
        直接连不上（实测踩到：``InvalidStatus: HTTP 404``）。判据是 ``Upgrade`` 头。

        这里只做**只读静态文件**：路径规范化后必须落在 ``web_dir`` 内，杜绝 ../ 穿越。
        """
        upgrade = (request.headers.get("Upgrade") or "").lower()
        if "websocket" in upgrade:
            return None                     # 交回给 websockets 完成握手
        try:
            path = unquote(urlsplit(request.path).path)
            if path in ("/", ""):
                path = "/index.html"
            if path == "/health":
                body = json.dumps(self.stats(), ensure_ascii=False).encode()
                return Response(200, "OK", Headers({
                    "Content-Type": "application/json; charset=utf-8",
                    "Content-Length": str(len(body)),
                }), body)

            rel = path.lstrip("/")
            target = (self.web_dir / rel).resolve()
            root = self.web_dir.resolve()
            if root not in target.parents and target != root:
                return Response(403, "Forbidden", Headers({}), b"forbidden")
            if not target.is_file():
                return Response(404, "Not Found", Headers({}), b"not found")

            body = target.read_bytes()
            ctype = _MIME.get(target.suffix.lower()) or \
                mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            return Response(200, "OK", Headers({
                "Content-Type": ctype,
                "Content-Length": str(len(body)),
                # 开发期禁用缓存：改完刷新就能生效，不必让用户清缓存
                "Cache-Control": "no-store",
            }), body)
        except Exception as ex:                                   # noqa: BLE001
            logger.warning("静态页服务失败: %s", ex)
            return Response(500, "Internal Server Error", Headers({}), b"error")

    # ------------------------------------------------------------------
    # WS 主处理
    # ------------------------------------------------------------------

    async def _handler(self, ws):
        conn: Optional[Connection] = None
        peer = self._peer(ws)
        try:
            # ---- 握手：等 hello ----
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                await ws.send(error(E_BAD_MESSAGE, "expected hello"))
                return
            if isinstance(raw, (bytes, bytearray)):
                await ws.send(error(E_BAD_MESSAGE, "expected hello (text) first"))
                return
            msg_type, fields = decode(raw)
            if msg_type != C_HELLO:
                await ws.send(error(E_BAD_MESSAGE, f"expected hello, got {msg_type!r}"))
                return
            if int(fields.get("version") or 0) != PROTOCOL_VERSION:
                await ws.send(error(
                    E_BAD_VERSION,
                    f"服务端协议版本 {PROTOCOL_VERSION}，客户端 {fields.get('version')}"))
                await ws.send(bye("version mismatch"))
                return
            if self.token and str(fields.get("token") or "") != self.token:
                await ws.send(error(E_UNAUTHORIZED, "token 不正确"))
                await ws.send(bye("unauthorized"))
                logger.warning("远端鉴权失败 peer=%s", peer)
                return

            session_id = f"phone:{self._random_id()}"
            conn = Connection(ws, session_id, self)
            self._conns[session_id] = conn
            # 显式绑定（优先于 session_prefix）—— 解绑后注册表会回落到本机端。
            # 只靠前缀的话，连接断开后"已消失的会话"仍会被路由到本端（实测发现）。
            if self.sts.transport_registry is not None:
                self.sts.transport_registry.bind(session_id, self)
            self.sts.vad.set_session_data(session_id, "user_id", session_id, create=True)
            # 远端按需唤醒：手机端在没接入本地唤醒词之前，音频会因为"没有唤醒词"
            # 被 pipeline 丢掉（`KWS: no wake word, skipping cloud STT`），用户看到
            # 的是"按住说了没反应"。"按住说话"这个手势本身就是明确的说话意图，
            # 唤醒词的防误触作用已由它承担（实测踩到）。
            self.sts._wake_on_connect[session_id] = True
            self._sink_tasks[session_id] = asyncio.create_task(self._sink(conn))
            await ws.send(ready(session_id))
            logger.info("远端已连接 %s peer=%s（当前 %d 条）", session_id, peer, len(self._conns))

            # ---- 主循环：收上行 ----
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    await self._on_audio(conn, bytes(raw))
                else:
                    stop = await self._on_control(conn, raw)
                    if stop:
                        break
        except websockets.exceptions.ConnectionClosed:
            logger.info("远端断开 %s", conn.session_id if conn else peer)
        except Exception as ex:                                   # noqa: BLE001
            logger.error("远端连接异常 %s: %s", conn.session_id if conn else peer, ex,
                         exc_info=True)
        finally:
            if conn is not None:
                await self._cleanup(conn, peer)

    async def _cleanup(self, conn: Connection, peer: str):
        conn.closed = True
        self._conns.pop(conn.session_id, None)
        self.forget(conn.session_id)                    # 基类：清掉录音窗/播放态
        if self.sts.transport_registry is not None:
            self.sts.transport_registry.unbind(conn.session_id)
        task = self._sink_tasks.pop(conn.session_id, None)
        if task is not None:
            task.cancel()
        pump = self._pump_tasks.pop(conn.session_id, None)
        if pump is not None:
            pump.cancel()
        logger.info("远端已清理 %s peer=%s（剩 %d 条）", conn.session_id, peer, len(self._conns))

    # ------------------------------------------------------------------
    # 上行
    # ------------------------------------------------------------------

    async def _on_audio(self, conn: Connection, frame: bytes):
        kind, _seq, pcm = unpack_audio(frame)
        if kind != BIN_UPLINK or not pcm:
            return
        conn.last_uplink_at = time.monotonic()
        if conn.first_uplink_at is None:
            # 本轮"用户开口时刻" —— 服务端时延的起点（只在第一块时置位）
            conn.first_uplink_at = time.monotonic()
        self.note_uplink(conn.session_id)               # 基类：录音窗判定
        conn.offer(conn.uplink, pcm)
        # 懒启动上行泵：**不要**只在收到 `start` 时才启动 —— 客户端可能先推流
        # 后补 start（或干脆不发 start）。曾经的实现导致音频只堆在队列里永远不处理
        # （实测：队列长度 1、pipeline 一块都没收到，表现为"手机说话没反应"）。
        self._pump(conn)

    async def _on_control(self, conn: Connection, raw: str) -> bool:
        msg_type, fields = decode(raw)
        if msg_type is None:
            await self._send_text(conn, error(E_BAD_MESSAGE, "无法解析的控制帧"))
            return False
        if msg_type == C_START:
            conn.speaking = True
            conn.first_uplink_at = None
            self._pump(conn)                            # 确保再喂协程在跑
            await self._push_state(conn)
        elif msg_type == C_END:
            conn.speaking = False
            await self._push_state(conn)
        elif msg_type == C_CANCEL:
            conn.speaking = False
            self._drop_queues(conn)
            await self.sts.stop_response(conn.session_id, None)
            await self._push_state(conn)
        elif msg_type == C_PING:
            await self._send_text(conn, pong(fields.get("t")))
        else:
            await self._send_text(conn, error(E_BAD_MESSAGE, f"未知控制帧 {msg_type!r}"))
        return False

    def _pump(self, conn: Connection):
        """启动"上行泵"：把队列里的音频喂给 pipeline。

        为什么用队列而不是直接在读协程里 await ``feed_audio``：``feed_audio`` 会
        触发 VAD/STT/LLM 一连串动作，一旦它在某处被 await 住，**网络读也会跟着停**
        —— 上行就会积压、时延飙升。队列把"收"和"处理"解耦（本机 adapter 的同类
        修复见 `低时延规划.md` 4.3）。
        """
        task = self._pump_tasks.get(conn.session_id)
        if task is not None and not task.done():
            return
        self._pump_tasks[conn.session_id] = asyncio.create_task(self._uplink_loop(conn))

    async def _uplink_loop(self, conn: Connection):
        try:
            while not conn.closed:
                pcm = await conn.uplink.get()
                t0 = time.monotonic()
                try:
                    await self.sts.feed_audio(pcm, conn.session_id)
                except Exception as ex:                           # noqa: BLE001
                    logger.error("上行喂入失败 %s: %s", conn.session_id, ex)
                    continue
                # 单块处理耗时若接近分块时长(32ms)，说明服务端跟不上实时
                cost = (time.monotonic() - t0) * 1000
                if cost > 60 and self.debug:
                    logger.warning("上行处理偏慢 %s: %.0fms/块", conn.session_id, cost)
        except asyncio.CancelledError:
            raise
        except Exception as ex:                                   # noqa: BLE001
            logger.error("上行循环异常 %s: %s", conn.session_id, ex)

    # ------------------------------------------------------------------
    # 下行
    # ------------------------------------------------------------------

    async def _sink(self, conn: Connection):
        """发送协程：先播"本轮 TTS"，再播"主动播报"。

        优先级：用户轮次 > 播报。因为播报是异步结果（等多久都还能再说），而本轮
        回复用户正等着听 —— 让播报插到前面会明显打断对话。
        """
        try:
            while not conn.closed:
                item = await self._next_downlink(conn)
                if item is None:
                    continue
                kind, payload = item
                if kind == "audio":
                    seq = conn.downlink_seq
                    conn.downlink_seq = (conn.downlink_seq + 1) & 0xFFFF
                    frame = pack_audio(payload, seq, downlink=True)
                    conn.playing = True
                    if conn.sent_first_audio_at is None:
                        conn.sent_first_audio_at = time.monotonic()
                        self._record_first_audio(conn)
                    await conn.ws.send(frame)
                elif kind == "text":
                    await self._send_text(conn, payload)
                elif kind == "end":
                    conn.playing = False
                    conn.sent_first_audio_at = None
                    conn.first_uplink_at = None
                    await self._push_state(conn)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as ex:                                   # noqa: BLE001
            logger.error("下行发送异常 %s: %s", conn.session_id, ex)

    async def _next_downlink(self, conn: Connection):
        """取下一项要发的东西：优先本轮音频/文本，其次播报。

        返回 ``(kind, payload)``，kind ∈ {"audio", "text", "end"}。
        """
        if not conn.downlink.empty():
            return conn.downlink.get_nowait()
        if not conn.announce.empty():
            return await conn.announce.get()
        # 两条都空 → 等任意一条就绪（不要忙等）
        getters = [asyncio.create_task(conn.downlink.get()),
                   asyncio.create_task(conn.announce.get())]
        try:
            done, pending = await asyncio.wait(getters, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                return t.result()
            return "end", None
        finally:
            for t in getters:
                if not t.done():
                    t.cancel()

    async def _send_text(self, conn: Connection, payload: str):
        try:
            await conn.ws.send(payload)
        except websockets.exceptions.ConnectionClosed:
            conn.closed = True

    #: 轮次结果 → 给用户看的话。**绝不能空着**（空着 = 用户以为助手死了）。
    _OUTCOME_TEXT = {
        N_NO_SPEECH: "没听清，请再说一次",
        N_NO_MATCH: "这轮没能回答上来，请再说一次",
        N_DELEGATED: "已转后台处理，结果稍后播报",
        N_ERROR: "这轮出错了",
    }

    async def _on_transcript(self, session_id: str, text: str):
        """把识别结果推给客户端，让用户看到"我听成了什么"。"""
        conn = self._conns.get(session_id)
        if conn is None or conn.closed or not text:
            return
        conn.offer(conn.downlink, ("text", text_message(S_STT, text)))

    async def _on_turn_outcome(self, session_id: str, outcome: str, text: str = ""):
        """pipeline 上报的"本轮结果" → 推给客户端。

        这一步是"永不静默"的兜底：无论正常回答、转后台、没听清还是出错，
        客户端都会收到一条明确的提示，而不是无限等下去（实测踩到过整轮静默，
        用户看到的是界面永远停在"稍等…"）。
        """
        conn = self._conns.get(session_id)
        if conn is None or conn.closed:
            return
        if outcome == N_DELEGATED:
            # 委托类已经有确认话术在播，不必再刷提示
            conn.offer(conn.downlink, ("text", turn("end", outcome=outcome)))
        elif outcome not in ("ok",):
            msg = text or self._OUTCOME_TEXT.get(outcome, "这轮没成功")
            conn.offer(conn.downlink, ("text", notice(outcome, msg)))
            conn.offer(conn.downlink, ("text", turn("end", outcome=outcome)))
        else:
            conn.offer(conn.downlink, ("text", turn("end", outcome=outcome)))
        # 轮次结束 → 状态几乎肯定变了（txn 已清），主动推一次
        self._schedule_state_push(conn)

    def _drop_queues(self, conn: Connection):
        for q in (conn.downlink, conn.announce):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    # ------------------------------------------------------------------
    # AudioTransport 接口（pipeline 通过这些方法驱动本端）
    # ------------------------------------------------------------------

    async def handle_response(self, response: STSResponse):
        """pipeline 的响应 → 发队列（**不阻塞**，满了丢最老）。"""
        conn = self._conns.get(response.session_id)
        if conn is None or conn.closed:
            return
        rtype = getattr(response, "type", None)
        if rtype == "chunk":
            # ⚠️ **音频块不携带文本**：同一个句子会被合成成多个音频块，若每块都挂文本，
            # 客户端会把同一句话打印 N 遍（实测：一句回复刷了 20 行）。
            # 文本统一由 `final` 给一次；期间的进度反馈用 STT/turn 事件。
            if response.audio_data:
                # ⚠️ 队列里所有元素都必须是 (kind, payload) 元组 —— 曾经这里直接塞裸
                # PCM 字节，与公告/文本路径的元组约定不一致，导致发送协程 unpack 崩掉
                # （`too many values to unpack`），表现为"客户端一直收不到音频"。
                if not conn.offer(conn.downlink, ("audio", response.audio_data)):
                    if self.debug:
                        logger.warning("下行丢帧 %s（队列满）", conn.session_id)
        elif rtype == "final":
            conn.offer(conn.downlink, ("end", None))
            # 把本轮完整回复也告诉客户端（有些路径 final 才有完整文本）
            if getattr(response, "text", None):
                conn.offer(conn.downlink, ("text", text_message(S_TEXT, response.text)))
            # ⚠️ **一轮结束必须重推 state**（实测踩到软锁）：
            # 用户"松开"那一刻正好是 `_active_txn` 非空的瞬间，服务端会回
            # `idle=false`（这是对的：正在为你处理这一轮）。但若之后再没有任何
            # state 推送，客户端就永远停在"稍等…"，**按钮再也按不动**。
            # 所以每轮收尾都主动广播一次最新状态。
            self._schedule_state_push(conn)

    async def emit(self, text: str, session_id: str = None):
        """主动播报（委托进度/结果）→ 单独队列，且**由服务端合成为音频**。

        委托结论是异步来的，手机可能已经在别的对话里；所以它走公告队列而不是
        插进本轮音频，客户端可以据此决定用不同提示音。
        """
        conn = self._conns.get(session_id)
        if conn is None or conn.closed or not text:
            return
        conn.offer(conn.announce, ("text", text_message(S_ANNOUNCE, text)))
        # 合成音频（与本机路径同样的 TTS 链路），失败就只发文本
        if getattr(self.sts, "text_only", False):
            return
        try:
            first = True
            async for audio in self.sts.tts.synthesize_stream(text):
                if conn.closed:
                    return
                if first:
                    conn.announce_playing = True
                conn.offer(conn.announce, ("audio", audio))
                first = False
            conn.offer(conn.announce, ("end", None))
        except Exception as ex:                                   # noqa: BLE001
            logger.error("播报合成失败 %s: %s", session_id, ex)

    async def _stop_playback(self, session_id: str = None) -> bool:
        """通知客户端停止播放，并清空待发音频。返回"是否真的中断了播放"。"""
        conn = self._conns.get(session_id)
        if conn is None:
            return False
        was_playing = conn.playing
        self._drop_queues(conn)
        conn.playing = False
        await self._push_state(conn)
        return was_playing

    async def _send_response(self, response):                     # pragma: no cover
        """基类接口占位；本类改为走发送队列（见 handle_response）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 状态与统计
    # ------------------------------------------------------------------

    async def _push_state(self, conn: Connection):
        idle = self.sts._is_idle(conn.session_id)
        await self._send_text(conn, state(idle, conn.playing, conn.speaking))

    def _schedule_state_push(self, conn: Connection, delay: float = 0.15):
        """稍后推一次 state（用于"一轮结束"这类需要让客户端恢复按动的时刻）。

        为什么要延迟：``final`` 是在 invoke 收尾时发出的，此刻
        ``_active_txn`` 可能还没被清掉（pipeline 在 yield final 之后才删它）。
        立刻推会读到"仍然忙"，等于没修。延迟一点再读才是真实状态。
        """
        async def later():
            try:
                await asyncio.sleep(delay)
                if not conn.closed:
                    await self._push_state(conn)
            except asyncio.CancelledError:
                raise
            except Exception as ex:                               # noqa: BLE001
                logger.warning("推送状态失败 %s: %s", conn.session_id, ex)

        asyncio.create_task(later())

    def is_active(self, session_id: str = None) -> bool:
        """本端此刻是否真的承载该会话（没有连接 = 不活跃）。

        这个方法关乎**播报能否开口**：注册表的仲裁会问"有没有远端在说话"，
        若把已断开的会话也算活跃，主动播报（委托进度/结果）会永远不播 ——
        一种完全静默的失效。
        """
        if session_id is None:
            return any(not c.closed for c in self._conns.values())
        conn = self._conns.get(session_id)
        return conn is not None and not conn.closed

    def is_recording(self, session_id: Optional[str] = None) -> bool:
        """本端是否在采集语音：由客户端上次推流的时间窗决定（基类实现）。"""
        if not self.is_active(session_id):
            return False
        return super().is_recording(session_id)

    def is_playing(self, session_id: Optional[str] = None) -> bool:
        """本端是否正在给该客户端送音频 —— 远端"忙不忙"的**唯一正确判据**。

        为什么不看 `is_recording`（实测踩到软锁）：客户端按下按钮时会先发 `start`，
        若服务端把"用户在说话"算进 idle 就会立刻回 `idle=false`，客户端据此显示
        "服务端还忙着"并拒绝下一次输入 —— 它用"自己在说话"这个事实把自己锁死了。
        远端要问的是"我这边是否正在放音、有没有进行中的轮次"，而不是"我是否在说话"。
        """
        if session_id is None:
            return any(c.playing for c in self._conns.values() if not c.closed)
        conn = self._conns.get(session_id)
        return bool(conn and not conn.closed and conn.playing)

    def stats(self) -> dict:
        return {
            "ok": True,
            "protocol": PROTOCOL_VERSION,
            "sampleRate": SAMPLE_RATE,
            "frameBytes": FRAME_BYTES,
            "tls": self.tls,
            "tlsFingerprint": self.tls_fingerprint,
            "connections": len(self._conns),
            "sessions": list(self._conns.keys()),
            "authRequired": bool(self.token),
            "speaking": [c.session_id for c in self._conns.values() if c.speaking],
        }

    def _record_first_audio(self, conn: Connection):
        if self.perf is None:
            return
        origin = conn.first_uplink_at
        if origin is None:
            return
        self.perf.record(
            session=conn.session_id,
            uplink_to_first_audio_ms=round((conn.sent_first_audio_at - origin) * 1000),
        )

    def _random_id(self) -> str:
        import secrets
        return secrets.token_hex(4)

    @staticmethod
    def _peer(ws) -> str:
        """客户端标识（优先取反代/隧道透传的真实 IP）。

        为什么必须看 `X-Forwarded-For`：走 cpolar 这类隧道（或任何反向代理）时，
        TCP 连接是从**本机**发起的，`remote_address` 恒定是 `127.0.0.1`
        —— 日志里就再也分不出"谁连上来了"，排查公网问题时等于瞎眼。
        代理一般会把真实来源放进 `X-Forwarded-For`（最左侧是第一跳）。
        """
        try:
            hdrs = getattr(getattr(ws, "request", None), "headers", None)
            if hdrs is not None:
                xff = hdrs.get("X-Forwarded-For") or hdrs.get("X-Real-IP")
                if xff:
                    return f"{xff.split(',')[0].strip()}(via proxy)"
        except Exception:                                         # noqa: BLE001
            pass
        try:
            addr = ws.remote_address
            return f"{addr[0]}:{addr[1]}" if addr else "?"
        except Exception:                                         # noqa: BLE001
            return "?"
