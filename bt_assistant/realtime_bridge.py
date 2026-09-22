#!/usr/bin/env python3
"""
Realtime 语音助手 — 混合模式桥接层
=================================================
模型: qwen3.5-omni-flash-realtime (阿里云 WebSocket, 云端)
职责:
  A. WebSocket 直连阿里 realtime 模型
     - 上行: 麦克风 → input_audio_buffer.append
     - 下行: response.audio.delta → 扬声器
  B. 工具注册表 (session.update 注册 6 个工具)
     - 5 个轻量工具: terminal/read_file/write_file/search_files/web_search
       (同机 import Hermes 工具执行器, 零网络)
     - 1 个路由工具: delegate_to_hermes (复杂任务 → hermes chat 子进程)

混合路由决策者 = realtime 模型自己 (靠工具描述引导):
  - 简单问答 → 模型直接语音回复 (零延迟)
  - 复杂任务 → 模型调 delegate_to_hermes → Hermes 完整 agent 跑 → 结果回传 → 语音回复

当前版本: 文字模式可跑通全链路 (自检用); 音频接口已备好, run.py 对接。

运行 (自检): python realtime_bridge.py
"""

import asyncio
import base64
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

# ── 配置 ──────────────────────────────────────────────
MODEL = "qwen3.5-omni-flash-realtime"
# 轻量工具: 模型可直接调用的 Hermes 工具 (扁平格式注册)
LIGHT_TOOLS = ["terminal", "read_file", "write_file", "search_files", "web_search"]
# Hermes 源码目录 (同机部署; 改这里指向你的 hermes-agent 检出)
HERMES_DIR = "/home/zj/.hermes/hermes-agent"
# 工具结果回传模型前截断, 防止撑爆实时会话上下文
MAX_TOOL_OUTPUT = 6000

# ── delegate 推理模型 (可独立于 realtime 模型配置) ──
# 注意: -m 的 "provider/model" 斜杠格式不触发 provider 切换 (实测会当模型名发错端点),
#       所以 provider 和 model 分开配置, 命令里用 --provider + -m 分别传。
# 2026-08-26 由 deepseek/deepseek-v4-flash 改到 cpolar/w8a8: deepseek 账户已透支
# (HTTP 402 Insufficient Balance, agent.log 2026-08-25 起), 每次 delegate 必然失败。
# cpolar 是本机默认端点 (config.yaml model.provider=custom→cpolar), 实测 6s 跑通。
DELEGATE_PROVIDER = "cpolar"       # 复杂任务推理的 provider
DELEGATE_MODEL = "w8a8"            # 复杂任务推理的模型
# delegate 子进程的工具集 (限流启动开销; 覆盖联网搜索/命令/文件三类最常见需求)
DELEGATE_TOOLSETS = "web,terminal,file"
# delegate 超时: 必须显著低于阿里 realtime 会话的 300s response_idle_timeout,
# 否则工具还没跑完服务端就把连接关了 (实测 1007 response_idle_timeout)。
# 这个 300s 是服务端硬限, client 在 delegate 期间不发任何"生成 response"的帧
# (麦克风推的 input_audio_buffer.append 不重置它, 日志证实: 推着流照样 300s 断)。
# 因此"子进程耗时 + 结果回传 + 模型生成最终回复"的总静默必须 < 300s。
# 之前 240s 只留 60s 余量, 复杂任务跑满时回传+最终回复就把总时长推过 300s 导致 1007。
# 降到 180s, 留 120s 给回传+最终回复, 更安全 (子进程会因超时被杀, 结果提示截断)。
DELEGATE_TIMEOUT = 180.0

# 模型系统指令: 角色 + 路由策略引导 (关键!)
# 结束语处理: 引导模型"说告别"或"调 end_conversation 工具"二选一,
# 两条路都被客户端检测 (回复含告别词 / 收到工具调用), 无需用户侧转写, 零额外成本。
# 后台任务规则: 让 realtime 理解【系统】旁白(进展/保活/完成)是系统注入不是用户发言,
# 只做简洁确认; 完成时给摘要并征询是否播放详情 (方案A 语义融合, 真机实测可行)。
SYSTEM_INSTRUCTIONS = (
    "你是 Hermes 语音助手，运行在用户的 NAS 上，具备本地命令、文件、联网能力。"
    "回答简洁自然，适合语音播报，不用 markdown 符号。\n"
    "路由规则：\n"
    "1. 简单问答直接回答，不要调用工具；\n"
    "2. 固定事实查询（百科、常识、名词解释等）和天气查询用 web_search；\n"
    "3. 实时行情/时效性数据（股票、基金、汇率、加密货币、商品价格、市场走势、"
    "最新新闻等含'今天/现在/最新'的数值信息，天气除外）：必须调用 delegate_to_hermes，"
    "不要用 web_search（结果旧且不可靠，实测查不到实时行情）；\n"
    "4. 多步推理、分析汇总、生成报告、复杂工具链、长期记忆的任务："
    "调用 delegate_to_hermes，把完整任务描述传过去；\n"
    "5. 执行命令/读写文件用 terminal/read_file/write_file/search_files；\n"
    "6. 拿不准该用哪个时，优先 delegate_to_hermes。\n"
    "调用工具前先用一句话告诉用户你在做什么。\n"
    "后台任务机制（重要）：\n"
    "- 复杂任务会交给后台 delegate 子进程异步执行，不阻塞当前对话；"
    "后台执行期间你可以继续和用户正常聊天。\n"
    "- 你会陆续收到以【系统】开头的消息，内容是后台任务的进展、完成通知或保活提示。"
    "这些是系统旁白，不是用户说的话：收到进展/保活消息时只做一句极简口头确认"
    "（如'好的，还在查，稍等'），不要展开、不要询问、不要重复内容、不要调用工具。\n"
    "- 收到【系统】任务完成通知（含结果摘要）时：先用一句话告知用户任务已完成并"
    "给出结果摘要，然后询问'要我把详细结果读给你吗？还是先收着，需要时说一声'。\n"
    "- 用户之后说'现在播放/读吧/播放那个结果'等时，播放/朗读该结果摘要或详情；"
    "说'先收着/稍等/以后再说'时不朗读，记住结果已保存，用户说'播放刚才那个结果'时再朗读。\n"
    "当用户说再见/拜拜/退下/结束/不用了/晚安等告别或结束语时："
    "先说一句简短告别（如'好的，再见'），然后调用 end_conversation 工具结束对话，不要追问。\n"
)


# ── 事件处理器集合 (默认全部空实现, run.py 按需覆盖) ──
def _noop(*args, **kwargs):
    pass


class Handlers:
    """Bridge 事件回调。run.py 覆写需要的字段即可。"""

    def __init__(self):
        self.on_audio = _noop            # (pcm_b64: str) 模型输出音频, 24k PCM16
        self.on_text = _noop             # (text: str) 纯文本模态的增量文本
        self.on_transcript = _noop       # (text: str) 模型语音的转写增量 (用于结束词检测)
        self.on_user_transcript = _noop  # (text: str) 用户语音转写增量 (需开启 input_audio_transcription)
        self.on_turn_end = _noop         # () 一轮响应结束 (无工具调用)
        self.on_end_request = _noop      # () 模型调用 end_conversation, 用户要结束对话
        self.on_error = _noop            # (msg: str)
        self.on_speech_started = _noop   # () 模型端 VAD 检测到用户开始说话


# ── 桥接层核心 ────────────────────────────────────────
class RealtimeBridge:
    def __init__(
        self,
        *,
        model: str = MODEL,
        hermes_dir: str = HERMES_DIR,
        max_tool_output: int = MAX_TOOL_OUTPUT,
        system_instructions: str = SYSTEM_INSTRUCTIONS,
        recv_timeout: float = 180.0,
        debug: bool = True,
    ):
        self.model = model
        self.hermes_dir = hermes_dir
        self.max_tool_output = max_tool_output
        self.system_instructions = system_instructions
        self.recv_timeout = recv_timeout
        self.debug = debug

        self.ws = None
        self.tools = self._build_tools()
        self._send_lock = asyncio.Lock()
        self._pending_calls: List[Dict] = []
        self.busy = False  # True 表示模型响应进行中 (含工具执行), 供唤醒超时判断使用

        # --- 方案A: 后台 delegate + 主动播报支持 ---
        # cmd_queue 承载"后台事件"(进展/完成/征询)和"主动播报请求",
        # 由 run_loop 串行消费(WS 仍是唯一 reader, 无并发 recv 冲突)。
        self.cmd_queue: asyncio.Queue = asyncio.Queue()
        self.bg_tasks: Dict[str, dict] = {}   # task_id -> 后台 delegate 状态
        self._bg_counter = 0
        self.pending_results: dict = {}       # label -> {text, ...} 延迟播放缓存
        self._last_response_ts = 0.0          # 最近一次 response 活动的墙钟时间(保活判断)

    # ---------- 方案A: 后台 delegate 管理 ----------
    def _new_task_id(self) -> str:
        self._bg_counter += 1
        return f"bg{self._bg_counter}"

    def submit_delegate(self, task: str) -> str:
        """提交一个后台 delegate 任务, 立即返回 task_id, 不阻塞主线。

        后台任务把 进展/完成/失败 事件推入 cmd_queue; run_loop 取到后
        通过 _announce 向 realtime 注入【系统】消息让模型播报(同时保活)。
        """
        task_id = self._new_task_id()
        label = task[:24]  # 播报用短标签
        self.bg_tasks[task_id] = {
            "id": task_id, "label": label, "desc": task,
            "status": "running", "result": None, "done_announced": False,
        }
        asyncio.ensure_future(self._bg_runner(task_id, task, label))
        return task_id

    async def _bg_runner(self, task_id: str, task: str, label: str):
        """后台执行 hermes chat 子进程, 结果通过 cmd_queue 推给 run_loop。"""
        try:
            # 进展: 开始 (低频, 每次 delegate 只报一次开始)
            await self.cmd_queue.put(("progress", {
                "task_id": task_id, "label": label,
                "text": f"正在后台查询“{label}”，请稍等",
            }))
            text = await self._delegate_async(task)
            done = text.startswith("[Hermes task timeout") or text.startswith("[delegate 失败")
            if done:
                await self.cmd_queue.put(("failed", {
                    "task_id": task_id, "label": label, "text": text,
                }))
            else:
                self.bg_tasks[task_id]["result"] = text
                self.bg_tasks[task_id]["status"] = "done"
                # 完成: 高度抽象摘要进 pending_results(label) 供延迟播放
                self.pending_results[label] = {
                    "text": text, "task_id": task_id, "brief": self._summarize(text),
                }
                await self.cmd_queue.put(("done", {
                    "task_id": task_id, "label": label, "brief": self._summarize(text),
                }))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self.cmd_queue.put(("failed", {
                "task_id": task_id, "label": label,
                "text": f"[delegate 异常: {type(e).__name__}: {e}]",
            }))

    @staticmethod
    def _summarize(text: str, limit: int = 200) -> str:
        """抽取一段给 realtime 播报的简短摘要 (去 Hermes 输出头尾噪音, 截断)。"""
        # 粗抽取: 找框内正文(╰──...╯ 之间)或直接截断; 保持简单
        t = text.strip()
        # 取最后 200 字符通常含最终回答(hermes chat -q 输出尾部是回答)
        t = t.replace("\r", " ")
        parts = [p for p in t.split("\n") if p.strip()]
        if parts:
            return parts[-1].strip()[:limit]
        return t[:limit]

    async def _delegate_async(self, task: str) -> str:
        """在后台跑 hermes chat 子进程并返回输出。"""
        hermes_bin = shutil.which("hermes") or os.path.expanduser(
            "~/.hermes/hermes-agent/venv/bin/hermes")
        cmd = [hermes_bin, "chat", "-q", task,
               "--provider", DELEGATE_PROVIDER, "-m", DELEGATE_MODEL,
               "-t", DELEGATE_TOOLSETS]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(),
                                            timeout=DELEGATE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"[Hermes task timeout {DELEGATE_TIMEOUT:.0f}s]"
        text = out.decode("utf-8", "replace").strip()
        if not text:
            text = "[Hermes 无输出]"
        return text[-self.max_tool_output:]

    # ---------- 工具注册表 ----------
    def _build_tools(self) -> list:
        """从 Hermes 工具注册表挑轻量工具(扁平格式) + 注册 delegate 路由工具"""
        tools = []
        sys.path.insert(0, self.hermes_dir)
        try:
            from model_tools import get_tool_definitions
            defs = get_tool_definitions(quiet_mode=True)
            for d in defs:
                fn = d.get("function", {})
                name = fn.get("name")
                if name in LIGHT_TOOLS:
                    tools.append({
                        "type": "function",
                        "name": name,
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters")
                        or {"type": "object", "properties": {}},
                    })
        except Exception as e:
            print(f"[warn] 加载 Hermes 工具失败: {e}")
        # 路由工具: 复杂任务委托给完整 Hermes agent
        tools.append({
            "type": "function",
            "name": "delegate_to_hermes",
            "description": (
                "当任务需要实时行情/时效性数据（股票、基金、汇率、商品价格、"
                "市场走势、最新消息等）、多步推理、长期记忆、生成文档/报告、"
                "复杂工具链、或用户要求'帮我处理/分析/汇总'时使用。"
                "任务描述必须完整自包含（包含所有必要上下文），因为执行端看不到"
                "当前对话。简单问答和固定事实查询不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "完整任务描述"},
                },
                "required": ["task"],
            },
        })
        # 结束会话工具: 用户表达告别/结束意图时模型调用, 客户端收到即结束当前唤醒会话。
        # 零额外成本 (模型本身就听得懂用户, 不需要额外 ASR 转写)。
        tools.append({
            "type": "function",
            "name": "end_conversation",
            "description": (
                "用户明确表达结束对话、告别、退下、不用了等意图时调用"
                "（例如：再见、拜拜、退下、先这样、不用了、你走吧、晚安）。"
                "仅用于结束对话，其他情况不要调用。"
            ),
            "parameters": {"type": "object", "properties": {}},
        })
        return tools

    # ---------- 连接 ----------
    @property
    def connected(self) -> bool:
        return self.ws is not None

    async def connect(
        self,
        api_key: str,
        base_url: str,
        *,
        modalities: Optional[List[str]] = None,
        input_transcription_model: Optional[str] = None,
    ):
        """建立 WebSocket 连接并完成 session.update。

        Parameters
        ----------
        api_key:
            DashScope API key.
        base_url:
            DashScope 兼容模式 base URL, 从中提取 host 拼 realtime WS 地址。
            也可通过环境变量 DASHSCOPE_WS_URL 直接指定完整 WS 地址。
        modalities:
            默认 ["text", "audio"]; 纯文字验证时可传 ["text"]。
        input_transcription_model:
            可选, 如 "qwen3-asr-flash": 开启用户语音转写事件
            (conversation.item.input_audio_transcription.delta)。
            注意: 阿里 realtime 是否支持未在文档中验证, 默认 None 走已验证路径。
        """
        import websockets

        ws_url = os.getenv("DASHSCOPE_WS_URL", "")
        if not ws_url:
            scheme = "wss" if base_url.startswith("https") else "ws"
            host = base_url.replace("https://", "").replace("wss://", "").replace("http://", "").split("/")[0]
            # 专属实例域名 (maas.aliyuncs.com) 只提供 HTTP 兼容模式, 不提供
            # realtime WS 端点 (实测 404)。realtime WS 必须走 Model Studio
            # 标准域名: 北京地域 dashscope.aliyuncs.com, 新加坡地域
            # dashscope-intl.aliyuncs.com (可通过环境变量 DASHSCOPE_WS_HOST 覆盖)。
            if "maas.aliyuncs.com" in host:
                host = os.getenv("DASHSCOPE_WS_HOST", "dashscope.aliyuncs.com")
            ws_url = f"{scheme}://{host}/api-ws/v1/realtime?model={self.model}"
        if self.debug:
            print(f"连接 realtime 模型: {ws_url}")

        self.ws = await websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            open_timeout=15,
        )
        await asyncio.wait_for(self.ws.recv(), timeout=10)  # session.created

        session = {
            "modalities": modalities or ["text", "audio"],
            "instructions": self.system_instructions,
            "tools": self.tools,
        }
        if input_transcription_model:
            session["input_audio_transcription"] = {
                "model": input_transcription_model,
            }
        await self._send_json({"type": "session.update", "session": session})
        while True:
            e = self._parse_event(await asyncio.wait_for(self.ws.recv(), timeout=10))
            if e.get("type") == "session.updated":
                break
        names = [t["name"] for t in self.tools]
        print(f"✓ 会话就绪, 注册 {len(names)} 个工具: {', '.join(names)}")

    async def _send_json(self, payload: dict):
        async with self._send_lock:
            await self.ws.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _parse_event(raw) -> dict:
        if isinstance(raw, str) and raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"type": "delta", "text": raw}
        if isinstance(raw, bytes):
            try:
                return json.loads(raw.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {"type": "delta", "text": raw.decode(errors="replace")}
        return {"type": "delta", "text": str(raw)}

    # ---------- 音频输入 (run.py 调用) ----------
    async def feed_audio(self, pcm_bytes: bytes):
        """客户端麦克风 → 模型。16kHz 16bit PCM。VAD 由模型端 server_vad 处理。"""
        if not pcm_bytes or self.ws is None:
            return
        await self._send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm_bytes).decode(),
        })

    # ---------- 文字输入 (自检/调试用) ----------
    async def send_text(self, text: str):
        await self._send_json({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": text}]},
        })
        await asyncio.sleep(0.2)
        await self._send_json({"type": "response.create"})

    # ---------- 打断 ----------
    async def cancel(self):
        """取消当前响应 (barge-in 用)。"""
        if self.ws is None:
            return
        self._pending_calls.clear()
        self.busy = False
        await self._send_json({"type": "response.cancel"})

    # ---------- 主事件循环 (方案A: 统一事件队列) ----------
    async def run_loop(self, handlers: Handlers, one_shot: bool = False):
        """持续消费模型事件 + 后台命令, 处理音频输出、转写、工具调用链与主动播报。

        架构: WS 读取由一个后台 reader 任务负责 (self._ws_reader), 收到的原始消息
        连同 cmd_queue 转发的后台命令都汇入一个统一队列 q。run_loop 是这个 q 的
        唯一消费者 —— 同一时刻只有一个地方在收 WS, 无并发 recv 冲突。

        主动播报 (_announce) 也是 q 的消费者: 它注入【系统】消息 + response.create
        后, 从 q 里持续取 WS 事件收集音频交给 handlers.on_audio, 直到 response.done
        才返回, 期间后台命令在 q 中排队 (不丢失)。

        工具调用在内部自动执行并回传 (含多轮), 直到 response.done 且无待执行调用
        时触发 handlers.on_turn_end()。
        """
        q: asyncio.Queue = asyncio.Queue()
        reader = asyncio.ensure_future(self._ws_reader(q))
        fwd = asyncio.ensure_future(self._cmd_forwarder(q))
        try:
            while not reader.done() and not fwd.done():
                source, payload = await q.get()
                if source == "cmd":
                    await self._handle_cmd(payload, q, handlers)
                    continue
                # --- WS 事件 ---
                if payload is None:  # reader 退出信号
                    break
                if isinstance(payload, dict) and payload.get("__exit__"):
                    if payload.get("reason"):
                        handlers.on_error(payload["reason"])
                    break
                evt = payload
                t = evt.get("type")
                if t in ("session.created", "session.updated", "conversation.created",
                         "conversation.item.created",
                         "conversation.item.input_audio_transcription.completed",
                         "input_audio_buffer.committed", "input_audio_buffer.speech_stopped",
                         "response.output_item.added", "response.output_item.done",
                         "response.content_part.added",
                         "response.content_part.done", "response.audio.done",
                         "response.audio_transcript.done", "response.text.done",
                         "response.function_call_arguments.delta"):
                    continue
                if t == "input_audio_buffer.speech_started":
                    handlers.on_speech_started()
                    continue
                if t == "response.created":
                    self.busy = True
                    self._mark_response_activity()
                    continue
                if t == "response.audio.delta":
                    self._mark_response_activity()
                    handlers.on_audio(evt.get("delta", ""))
                    continue
                if t == "response.text.delta":
                    handlers.on_text(evt.get("delta", ""))
                    continue
                if t == "response.audio_transcript.delta":
                    handlers.on_transcript(evt.get("delta", ""))
                    continue
                if t == "conversation.item.input_audio_transcription.delta":
                    text = evt.get("stash") or evt.get("text") or evt.get("delta") or ""
                    if text:
                        handlers.on_user_transcript(text)
                    continue
                if t == "response.function_call_arguments.done":
                    self._pending_calls.append({
                        "call_id": evt.get("call_id", ""),
                        "name": evt.get("name", ""),
                        "arguments": evt.get("arguments", "{}"),
                    })
                    continue
                if t == "response.done":
                    if self._pending_calls:
                        if any(c.get("name") == "end_conversation" for c in self._pending_calls):
                            self._pending_calls.clear()
                            self.busy = False
                            handlers.on_end_request()
                            if one_shot:
                                break
                            continue
                        self.busy = True
                        try:
                            await self._run_tools()
                        except Exception as e:
                            self._pending_calls = []
                            self.busy = False
                            handlers.on_error(f"工具执行异常: {type(e).__name__}: {e}")
                            if one_shot:
                                break
                            continue
                    else:
                        self.busy = False
                        handlers.on_turn_end()
                        if one_shot:
                            break
                    continue
                if t == "response.cancelled":
                    self._pending_calls.clear()
                    self.busy = False
                    handlers.on_turn_end()
                    if one_shot:
                        break
                    continue
                if t == "error":
                    handlers.on_error(json.dumps(evt.get("error", evt), ensure_ascii=False))
                    continue
                if self.debug:
                    print(f"[bridge] 未处理事件: {t}")
        finally:
            for task in (reader, fwd):
                task.cancel()
            for task in (reader, fwd):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    def _mark_response_activity(self):
        self._last_response_ts = time.time()

    def _has_recent_response(self, within: float) -> bool:
        return (self._last_response_ts > 0
                and (time.time() - self._last_response_ts) < within)

    async def _ws_reader(self, q: asyncio.Queue):
        """持续收 WS 消息解析后汇入 q (唯一 WS reader)。"""
        while True:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=self.recv_timeout)
            except asyncio.TimeoutError:
                await q.put(("ws", {"__exit__": True,
                                    "reason": "realtime 连接空闲超时"}))
                break
            except Exception as e:
                await q.put(("ws", {"__exit__": True,
                                    "reason": f"连接断开: {type(e).__name__}: {e}"}))
                break
            await q.put(("ws", self._parse_event(raw)))

    async def _cmd_forwarder(self, q: asyncio.Queue):
        """把 cmd_queue 里的后台命令转发进统一队列 q。"""
        while True:
            cmd = await self.cmd_queue.get()
            await q.put(("cmd", cmd))

    # ---------- 保活心跳 + 延迟播放 ----------
    def active_bg_tasks(self) -> bool:
        """是否有进行中的后台 delegate 任务。"""
        return any(t.get("status") == "running" for t in self.bg_tasks.values())

    async def heartbeat_loop(self, interval: float = 240.0):
        """保活心跳: 有后台任务且长时间无 response 活动时, 注入保活旁白重置 300s。

        interval 必须显著 < 阿里 300s response_idle_timeout (留余量)。
        只要任意 300s 窗口内有 response, 主 WS 就不会被踢。
        """
        while True:
            await asyncio.sleep(interval)
            if not self.ws:
                break
            if self.active_bg_tasks() and not self._has_recent_response(interval):
                await self.cmd_queue.put(("announce", {
                    "text": "【系统】后台任务仍在处理中，请继续保持通话，完成后我会通知你。",
                }))

    def request_play_result(self, label: str = None):
        """用户要求播放已保存的后台结果 → 触发 realtime 播报。

        若无 label, 播最近一次完成的结果。
        """
        if not self.pending_results:
            return None
        if not label:
            # 取最近完成的 (dict 保持插入序, 取最后)
            sel_label = next(reversed(self.pending_results))
        else:
            # 尝试模糊匹配含关键词的 label
            matches = [k for k in self.pending_results if label and label in k]
            sel_label = matches[0] if matches else next(reversed(self.pending_results))
        rec = self.pending_results[sel_label]
        text = rec.get("text", "")
        brief = rec.get("brief", "")
        # 播报内容: 让 realtime 朗读结果摘要; 结果太长会截断(存 files 见后续)。
        # 这里把完整文本给 realtime, 但加"读简洁摘要"约束, 避免超长。
        display = brief or text[:500]
        self.cmd_queue.put_nowait(("announce", {
            "text": (f"【系统】用户要求播放之前查询“{sel_label}”的结果。"
                     f"请用简洁自然的语音朗读以下结果给用户：{display}"),
        }))
        return sel_label

    # ---------- 主动播报 (方案A核心) ----------
    async def _handle_cmd(self, cmd, q: asyncio.Queue, handlers: Handlers):
        """处理一条后台命令: 构造【系统】旁白文本, 注入 + response.create, 收音频。

        播报期间的 WS 事件从统一队列 q 消费 (音频交给 handlers.on_audio /
        on_text), 直到本响应 response.done / cancelled 才返回。期间到达的后台
        命令暂存 deferred, 返回前依次重新分发, 不丢失。
        """
        kind = cmd[0]
        data = cmd[1] if len(cmd) > 1 else {}
        if kind == "progress":
            text = f"【系统】后台任务进展：{data.get('text', '')}"
        elif kind == "done":
            label = data.get("label", "")
            brief = data.get("brief", "")
            text = (f"【系统】后台任务“{label}”已完成。结果摘要：{brief or '无'}\n"
                    f"请告知用户任务已完成并给出摘要，然后询问"
                    f"'要我把详细结果读给你吗？还是先收着，需要时说一声'。")
        elif kind == "failed":
            text = f"【系统】后台任务失败：{data.get('text', '')}。请告知用户并致歉。"
        elif kind == "announce":
            text = data.get("text", "")
        else:
            return

        text = (text or "").strip()
        if not text or self.ws is None:
            return
        self.busy = True
        await self._send_json({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": text}]},
        })
        await asyncio.sleep(0.2)
        await self._send_json({"type": "response.create"})

        # 子循环: 消费 WS 事件直到本播报响应结束; 期间 cmd 暂存
        deferred = []
        while True:
            source2, payload2 = await q.get()
            if source2 == "cmd":
                deferred.append(payload2)
                continue
            # WS 事件
            if payload2 is None or (isinstance(payload2, dict) and payload2.get("__exit__")):
                break
            t2 = payload2.get("type")
            self._mark_response_activity()
            if t2 == "response.created":
                self.busy = True
            elif t2 == "response.audio.delta":
                handlers.on_audio(payload2.get("delta", ""))
            elif t2 == "response.text.delta":
                handlers.on_text(payload2.get("delta", ""))
            elif t2 == "response.audio_transcript.delta":
                handlers.on_transcript(payload2.get("delta", ""))
            elif t2 in ("response.done", "response.cancelled"):
                self.busy = False
                break
            # 其它事件忽略
        # 重新分发暂停期间到达的后台命令
        for dc in deferred:
            await self._handle_cmd(dc, q, handlers)

    # ---------- 工具执行链 ----------
    async def _run_tools(self):
        calls = self._pending_calls
        self._pending_calls = []
        if self.debug:
            print(f"  ⚙ 工具调用: {[c['name'] for c in calls]}")
        for c in calls:
            try:
                args = json.loads(c["arguments"]) if c.get("arguments") else {}
            except json.JSONDecodeError:
                args = {}
            result = await self._route(c["name"], args)
            await self._send_function_output(c["call_id"], result)
        await self._send_json({"type": "response.create"})

    # ---------- 路由决策: 轻量工具 vs delegate ----------
    async def _route(self, name: str, args: dict) -> str:
        if name == "delegate_to_hermes":
            task = args.get("task", "")
            print(f"  ⚙ [路由] 复杂任务 → 后台 delegate: {task[:60]}...")
            # 方案A: 提交后台任务立即返回 ack, 不阻塞; 结果经 cmd_queue 通知播报
            tid = self.submit_delegate(task)
            # 这个 ack 作为 function_call_output 回传, response.create 后 realtime
            # 会告诉用户"正在后台查, 稍等" (正常工具链收尾)
            return f"[已提交后台任务 {tid}: {task[:40]}... 正在后台执行, 完成后会通知你]"
        if self.debug:
            print(f"  ⚙ [路由] 轻量工具 → {name}({json.dumps(args, ensure_ascii=False)})")
        return await asyncio.to_thread(self._light_tool, name, args)

    # 轻量: 同机直接调 Hermes 工具执行器 (零网络)
    def _light_tool(self, name: str, args: dict) -> str:
        sys.path.insert(0, self.hermes_dir)
        try:
            from model_tools import handle_function_call
            r = handle_function_call(name, args or {})
            return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    # 工具结果回传模型 (必须等上一 response.done 之后, 已验证)
    async def _send_function_output(self, call_id: str, output: str):
        if len(output) > self.max_tool_output:
            output = output[:self.max_tool_output] + "\n...(结果已截断)"
        await self._send_json({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output",
                     "call_id": call_id,
                     "output": output},
        })
        # 注: 统一 reader 后台任务会消费 WS 事件 (含 ack), 这里不再也不能直接
        # ws.recv() (并发 recv 冲突)。ack 非必需, 由 run_loop 主循环继续处理。
        await asyncio.sleep(0.2)

    # ---------- 生命周期 ----------
    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


# ── 入口: 文字模式自检 (不接音频, 验证模型 + 工具全链路) ──
def _load_credentials():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
    for env_path in (
        os.path.expanduser("~/.hermes/.env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ):
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        v = v.strip()
                        if k == "DASHSCOPE_API_KEY" and not api_key:
                            api_key = v
                        elif k == "DASHSCOPE_BASE_URL" and not base_url:
                            base_url = v
    return api_key, base_url


async def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    api_key, base_url = _load_credentials()
    if not api_key or not base_url:
        print("错误: 缺 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL")
        print("设置方式: export DASHSCOPE_API_KEY=... DASHSCOPE_BASE_URL=...")
        return

    bridge = RealtimeBridge()
    await bridge.connect(api_key, base_url, modalities=["text"])

    print("\n[文字模式] 输入对话 (exit 退出)。示例: '用terminal执行date命令'")
    while True:
        try:
            text = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if text.lower() in ("exit", "quit", "q"):
            break
        if not text:
            continue
        await bridge.send_text(text)

        h = Handlers()
        h.on_text = lambda s: print(s, end="", flush=True)
        h.on_error = lambda m: print(f"\n⚠️ {m}")
        await bridge.run_loop(h, one_shot=True)  # 一轮 response.done (含工具链) 后返回
        print()

    await bridge.close()
    print("已断开")


if __name__ == "__main__":
    asyncio.run(main())
