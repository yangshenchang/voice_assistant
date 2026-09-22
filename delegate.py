"""复杂任务委托（delegate）客户端 + 播报调度。

对应 `bt_assistant/qqbot_relay`（常驻在 qqbot profile 内的中继插件）：
  POST /task        → 202 {taskId}        提交(立即返回, 异步执行)
  GET  /task/<id>   → {state, progress[], headline, full, qq...}
  POST /cancel      → 取消当前任务

本模块负责"什么时候把什么话播出来"这一层策略，把语音链路的其余部分
（决定何时开口、如何合成）留给 pipeline：

- **提交即走**：绝不阻塞当前轮次，否则首字时延会从几百毫秒拉到几十秒。
- **进度累计后择机播**："正在联网查资料 → 正在读网页"逐条播是噪音；且只在
  用户已等待超过阈值、且当前空闲时才播，避免打断用户。
- **结果择机播**：完成后播 headline（口语结论），并告知详情已发到 QQ。
- **可取消**：用户说"别查了"时中断后台任务。

注意：所有请求都必须 `trust_env=False` —— 本机开着代理，httpx 默认会把
127.0.0.1 的请求也发给代理，结果拿到 502。
"""

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

#: 工具名 → 口语化说法（进度播报不能念英文工具名）
_TOOL_PHRASES = {
    "web_search": "正在联网查资料",
    "web_fetch": "正在读网页",
    "read": "正在看文件",
    "glob": "正在翻目录",
    "grep": "正在搜内容",
}
_PROGRESS_FALLBACK = "还在处理"


class DelegateTask:
    """一个后台委托任务的状态。"""

    def __init__(self, task_id: str, task: str, session_id: str, context_id: str = None):
        self.id = task_id
        self.task = task
        self.session_id = session_id
        self.context_id = context_id
        self.state = "running"
        self.headline = ""
        self.full = ""
        self.qq_sent = False
        self.error: Optional[str] = None
        self.phrases: List[str] = []       # 口语化进度短语(去重保序)
        self.raw_progress = 0              # 已消费的原始进度条数
        self.announced_phrases = 0         # 已播报到的短语位置
        self.started_at = time.time()
        self.last_announce_at = 0.0
        self.finished_at: Optional[float] = None

    def absorb(self, progress: List[Dict]) -> bool:
        """吸收新进度，返回是否有新内容。"""
        new = progress[self.raw_progress:]
        self.raw_progress = len(progress)
        for p in new:
            phrase = phrase_for(p.get("text") or "")
            if phrase and phrase not in self.phrases:
                self.phrases.append(phrase)
        return bool(new)


def phrase_for(progress_text: str) -> str:
    """把中继的原始进度文本翻成适合朗读的一句话。"""
    text = progress_text or ""
    if text.startswith("正在使用 "):
        tool = text[len("正在使用 "):].split(" ")[0].strip()
        return _TOOL_PHRASES.get(tool, _PROGRESS_FALLBACK)
    if text.startswith("工具失败"):
        return ""                      # 单次工具失败会重试，不值得播报
    if text.startswith("本轮结束") or text.startswith("agent 错误"):
        return ""                      # 错误另行走 error 分支
    return ""


class DelegateManager:
    """管理 delegate 任务的生命周期与播报时机。

    Parameters
    ----------
    base_url:
        中继控制面地址（默认 http://127.0.0.1:8765）。
    announce:
        ``async (text, session_id) -> None`` —— 由 pipeline 提供，负责真正开口。
    is_idle:
        ``(session_id) -> bool`` —— 当前是否适合开口（没在放音、没在录音、没有进行中的轮次）。
    progress_after:
        用户等待超过这么多秒才播第一次进度。
    progress_interval:
        之后每隔这么多秒最多再播一次进度。
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8765",
        announce: Callable = None,
        is_idle: Callable = None,
        remember: Callable = None,
        enabled: bool = True,
        progress_after: float = 25.0,
        progress_interval: float = 45.0,
        poll_interval: float = 3.0,
        request_timeout: float = 10.0,
        task_timeout: float = 900.0,
        debug: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self._announce = announce
        self._is_idle = is_idle or (lambda sid: True)
        #: ``async (context_id, text) -> None``：把播报出去的结论写进对话历史。
        #: 不写的话会出现"助手刚说完结果、转头就说自己没收到结果"（实测确认过）。
        self._remember = remember
        self.enabled = enabled
        self.progress_after = progress_after
        self.progress_interval = progress_interval
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self.debug = debug

        # trust_env=False: 绕开本机代理（否则 localhost 请求会被代理拦成 502）
        self._client = httpx.AsyncClient(timeout=request_timeout, trust_env=False)
        self.tasks: Dict[str, DelegateTask] = {}
        self._watchers: Dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    async def submit(self, task: str, session_id: str = None,
                     context_id: str = None) -> Optional[str]:
        """提交任务，立即返回 task_id（失败返回 None，调用方可据此回话给用户）。"""
        if not self.enabled:
            return None
        try:
            r = await self._client.post(f"{self.base_url}/task", json={"task": task})
            r.raise_for_status()
            task_id = r.json()["taskId"]
        except Exception as ex:                                  # noqa: BLE001
            logger.error(f"委托提交失败: {type(ex).__name__}: {ex}")
            return None
        rec = DelegateTask(task_id, task, session_id, context_id)
        self.tasks[task_id] = rec
        self._watchers[task_id] = asyncio.create_task(self._watch(rec))
        logger.info(f"委托已提交 {task_id}: {task[:60]}")
        return task_id

    def cancel(self, session_id: str = None) -> Optional[str]:
        """请求取消当前任务（中继侧是单并发，取消的就是它正在跑的那个）。"""
        for rec in self.tasks.values():
            if rec.state == "running" and (session_id is None or rec.session_id == session_id):
                asyncio.create_task(self._cancel_remote())
                rec.state = "cancelled"
                rec.finished_at = time.time()
                logger.info(f"委托已取消 {rec.id}")
                return rec.id
        return None

    def running(self, session_id: str = None) -> List[DelegateTask]:
        return [
            r for r in self.tasks.values()
            if r.state == "running" and (session_id is None or r.session_id == session_id)
        ]

    async def close(self):
        for w in list(self._watchers.values()):
            w.cancel()
        self._watchers.clear()
        await self._client.aclose()

    # ------------------------------------------------------------------
    # 内部：轮询 + 播报决策
    # ------------------------------------------------------------------

    async def _cancel_remote(self):
        try:
            await self._client.post(f"{self.base_url}/cancel")
        except Exception as ex:                                   # noqa: BLE001
            logger.warning(f"取消请求失败: {ex}")

    async def _watch(self, rec: DelegateTask):
        """轮询任务状态，按策略播报进度与结果。"""
        deadline = rec.started_at + self.task_timeout
        try:
            while rec.state == "running":
                await asyncio.sleep(self.poll_interval)
                if rec.state != "running":
                    return
                if time.time() > deadline:
                    rec.state = "error"
                    rec.error = "超时"
                    logger.warning(f"委托 {rec.id} 超时")
                    break
                try:
                    r = await self._client.get(f"{self.base_url}/task/{rec.id}")
                    if r.status_code == 404:
                        continue
                    r.raise_for_status()
                    data = r.json()
                except Exception as ex:                           # noqa: BLE001
                    if self.debug:
                        logger.warning(f"轮询 {rec.id} 失败: {ex}")
                    continue

                rec.absorb(data.get("progress") or [])
                state = data.get("state")
                if state == "running":
                    await self._maybe_announce_progress(rec)
                    continue
                if state == "done":
                    rec.headline = (data.get("headline") or "").strip()
                    rec.full = (data.get("full") or "").strip()
                    rec.qq_sent = bool(data.get("qq")) and not data.get("qqError")
                    rec.state = "done"
                else:
                    rec.state = "error"
                    rec.error = data.get("error") or "委托执行失败"
                rec.finished_at = time.time()
                break
        except asyncio.CancelledError:
            raise
        except Exception as ex:                                   # noqa: BLE001
            rec.state = "error"
            rec.error = f"{type(ex).__name__}: {ex}"
            logger.error(f"委托 {rec.id} 监视异常: {ex}")

        if rec.state == "done":
            await self._announce_result(rec)
        elif rec.state == "error":
            await self._announce_text(
                rec, "刚才那个问题没查成，稍后你可以再让我试试。"
            )

    async def _maybe_announce_progress(self, rec: DelegateTask):
        waited = time.time() - rec.started_at
        if waited < self.progress_after:
            return
        if rec.last_announce_at and (time.time() - rec.last_announce_at) < self.progress_interval:
            return
        if not self._is_idle(rec.session_id):
            return                      # 用户正忙 → 等下一轮轮询再判

        # 累计后的最新状态：只播最新的一条，而不是逐条播
        fresh = rec.phrases[rec.announced_phrases:]
        rec.announced_phrases = len(rec.phrases)
        detail = fresh[-1] if fresh else _PROGRESS_FALLBACK
        rec.last_announce_at = time.time()
        await self._announce_text(rec, f"刚才那个问题还在查，{detail}。")

    async def _announce_result(self, rec: DelegateTask):
        text = rec.headline or "刚才那个问题有结果了。"
        if not text.startswith("刚才") and not text.startswith("顺便"):
            text = f"刚才那个问题有结果了：{text}"
        if rec.qq_sent:
            text = f"{text} 详细的我发到你 QQ 上了。"
        rec.last_announce_at = time.time()
        await self._announce_text(rec, text)
        # 把刚刚**说出去的话**写进对话历史 —— 否则用户追问"刚才那个结果是多少"时，
        # 前端模型完全不知道（实测：它会答"我还没收到后台结果"，而用户刚听到过）。
        if self._remember and rec.context_id:
            try:
                await self._remember(rec.context_id, text)
            except Exception as ex:                               # noqa: BLE001
                logger.error(f"写入委托结论到历史失败: {ex}")

    async def _announce_text(self, rec: DelegateTask, text: str):
        if not text or self._announce is None:
            return
        try:
            await self._announce(text, rec.session_id)
        except Exception as ex:                                   # noqa: BLE001
            logger.error(f"播报失败: {ex}")
