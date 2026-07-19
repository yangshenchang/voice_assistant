"""LLM service abstractions and the nanobot WebSocket client."""

import asyncio
import inspect
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from ..context_manager import ContextTracker
from ..models import LLMResponse, ToolCall

logger = logging.getLogger(__name__)


# ========================================================================
# Tool definition
# ========================================================================

class Tool:
    """A callable tool registered with the LLM service."""

    def __init__(
        self,
        name: str,
        spec: Dict,
        func: Callable,
        instruction: str = None,
        is_dynamic: bool = False,
    ):
        self.name = name
        self.spec = spec
        self.func = func
        self.instruction = instruction
        self.is_dynamic = is_dynamic


# ========================================================================
# Abstract LLM service
# ========================================================================

class LLMService(ABC):
    """Abstract base for LLM backends.

    Handles text splitting, voice-text extraction, tool routing,
    and delegates actual streaming to ``get_llm_stream_response``.
    """

    def __init__(
        self,
        *,
        system_prompt: str = None,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.5,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        context_tracker: ContextTracker = None,
        debug: bool = False,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.split_chars = split_chars or ["。", "？", "！", ". ", "?", "!"]
        self.option_split_chars = option_split_chars or ["、", ", "]
        self.option_split_threshold = option_split_threshold

        # Precompile split patterns
        patterns = []
        for c in self.option_split_chars:
            patterns.append(f"{re.escape(c)}" if c.endswith(" ") else f"{re.escape(c)}\\s?")
        self._opt_split_re = re.compile(
            f"({'|'.join(patterns)})\\s*(?!.*({'|'.join(patterns)}))"
        )

        self.voice_text_tag = voice_text_tag
        self.tools: Dict[str, Tool] = {}
        self.context_tracker = context_tracker or ContextTracker()
        self.debug = debug

        # Hooks
        self._request_filter: Callable = lambda text: text
        self._on_before_tool_calls: Callable = self._default_before_tool_calls

    # -- decorators -------------------------------------------------------

    def request_filter(self, func):
        self._request_filter = func
        return func

    def on_before_tool_calls(self, func):
        self._on_before_tool_calls = func
        return func

    async def _default_before_tool_calls(self, tool_calls: List[ToolCall]):
        pass

    # -- abstract methods -------------------------------------------------

    @abstractmethod
    async def compose_messages(
        self, context_id: str, text: str,
        files: List[Dict] = None, system_prompt_params: Dict = None,
    ) -> List[Dict]:
        """Build the message list to send to the LLM."""

    @abstractmethod
    async def update_context(
        self, context_id: str, messages: List[Dict], response_text: str,
    ):
        """Persist conversation history (no-op when nanobot manages it)."""

    @abstractmethod
    async def get_llm_stream_response(
        self, context_id: str, user_id: str,
        messages: List[Dict], system_prompt_params: Dict = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Yield LLMResponse chunks from the LLM."""

    # -- tool execution ---------------------------------------------------

    async def execute_tool(self, name: str, arguments: dict, metadata: dict = None):
        tool = self.tools[name]
        if "metadata" in inspect.signature(tool.func).parameters:
            arguments["metadata"] = metadata
        return await tool.func(**arguments)

    # -- main streaming entry point ---------------------------------------

    async def chat_stream(
        self,
        context_id: str,
        user_id: str,
        text: str,
        files: List[Dict] = None,
        system_prompt_params: Dict = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """High-level streaming entry point with text splitting & voice extraction."""
        logger.info(f"User: {text}")
        text = self._request_filter(text)

        if not text and not files:
            return

        messages = await self.compose_messages(context_id, text, files, system_prompt_params)
        msg_count_start = len(messages) - 1

        stream_buffer = ""
        response_text = ""
        in_voice = False
        tag_start = f"<{self.voice_text_tag}>" if self.voice_text_tag else None
        tag_end = f"</{self.voice_text_tag}>" if self.voice_text_tag else None

        def extract_voice(segment: str) -> Optional[str]:
            nonlocal in_voice
            if not tag_start:
                return self._remove_control_tags(segment)
            if tag_start in segment and tag_end in segment:
                in_voice = False
                return self._remove_control_tags(
                    segment[segment.find(tag_start) + len(tag_start):segment.find(tag_end)]
                )
            if tag_start in segment:
                in_voice = True
                return self._remove_control_tags(segment[segment.find(tag_start) + len(tag_start):])
            if tag_end in segment:
                if in_voice:
                    in_voice = False
                    return self._remove_control_tags(segment[:segment.find(tag_end)])
            if in_voice:
                return self._remove_control_tags(segment)
            return None

        async for chunk in self.get_llm_stream_response(context_id, user_id, messages, system_prompt_params):
            if chunk.tool_call:
                yield chunk
                continue

            stream_buffer += chunk.text or ""

            for sp in self.split_chars:
                stream_buffer = stream_buffer.replace(sp, sp + "|")

            if len(stream_buffer) > self.option_split_threshold:
                stream_buffer = self._replace_last_opt_split(stream_buffer)

            parts = stream_buffer.split("|")
            while len(parts) > 1:
                sentence = parts.pop(0)
                stream_buffer = "|".join(parts)
                voice = extract_voice(sentence)
                yield LLMResponse(context_id, sentence, voice)
                response_text += sentence
                parts = stream_buffer.split("|")

            await asyncio.sleep(0.001)

        if stream_buffer:
            voice = extract_voice(stream_buffer)
            yield LLMResponse(context_id, stream_buffer, voice)
            response_text += stream_buffer

        logger.info(f"AI: {response_text}")
        if len(messages) > msg_count_start:
            await self.update_context(
                context_id,
                messages[msg_count_start - len(messages):],
                response_text,
            )

    # -- helpers ----------------------------------------------------------

    def _replace_last_opt_split(self, text: str) -> str:
        return self._opt_split_re.sub(r"\1|", text)

    @staticmethod
    def _remove_control_tags(text: str) -> str:
        return re.sub(r"\[(\w+):([^\]]+)\]", "", text).strip()


# ========================================================================
# Nanobot WebSocket Service
# ========================================================================

class NanobotWebSocketService(LLMService):
    """LLM backend that streams responses from a **nanobot** gateway via WebSocket.

    Parameters
    ----------
    ws_url:
        nanobot WebSocket endpoint, e.g. ``ws://127.0.0.1:8765/``.
    token:
        Shared secret that matches nanobot's ``channels.websocket.token``.
    system_prompt:
        Prepended to the first message of each chat.
    model:
        Model name passed through to nanobot (default ``deepseek-v4-flash``).
    """

    def __init__(
        self,
        *,
        ws_url: str = "ws://127.0.0.1:8765/",
        token: str = None,
        system_prompt: str = None,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.5,
        split_chars: List[str] = None,
        voice_text_tag: str = None,
        context_tracker: ContextTracker = None,
        debug: bool = False,
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            split_chars=split_chars or ["。", "？", "！", ". ", "?", "!"],
            voice_text_tag=voice_text_tag,
            context_tracker=context_tracker or ContextTracker(),
            debug=debug,
        )
        self.ws_url = ws_url
        self.token = token

        # WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_lock = asyncio.Lock()
        self._recv_task: Optional[asyncio.Task] = None
        self._default_chat_id: Optional[str] = None

        # Per-chat multiplexing
        self._queues: Dict[str, asyncio.Queue] = {}
        self._pending_events: Dict[str, List[dict]] = {}
        self._attach_future: Optional[asyncio.Future] = None

        # litests context_id → nanobot chat_id
        self._chat_map: Dict[str, str] = {}
        self._system_prompt_sent: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _ensure_connection(self):
        """Connect or reconnect the WebSocket."""
        if self._ws is not None:
            try:
                await self._ws.ping()
                return
            except Exception:
                if self.debug:
                    logger.info("nanobot ping failed, reconnecting...")
                self._ws = None

        async with self._ws_lock:
            if self._ws is not None:
                return

            url = self.ws_url
            if self.token:
                sep = "?" if "?" not in url else "&"
                url = f"{url}{sep}token={self.token}"

            if self.debug:
                logger.info(f"Connecting nanobot: {url}")
            self._ws = await websockets.connect(url)

            # Wait for 'ready'
            raw = await self._ws.recv()
            evt = json.loads(raw)
            if evt.get("event") != "ready":
                logger.warning(f"Expected 'ready', got: {evt}")
            self._default_chat_id = evt.get("chat_id")
            logger.info(f"nanobot ready (chat_id={self._default_chat_id})")

            # Start recv loop
            if self._recv_task is None or self._recv_task.done():
                self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        """Single consumer routing frames to per-chat queues."""
        while True:
            try:
                raw = await self._ws.recv()
            except ConnectionClosed as e:
                logger.warning(f"nanobot closed: {e}")
                self._ws = None
                for q in self._queues.values():
                    await q.put(None)
                self._queues.clear()
                self._pending_events.clear()
                break
            except Exception as e:
                logger.error(f"nanobot recv error: {e}")
                continue

            evt = self._parse_event(raw)
            evt_type = evt.get("event")
            chat_id = evt.get("chat_id") or self._default_chat_id

            if evt_type == "attached":
                if self._attach_future and not self._attach_future.done():
                    self._attach_future.set_result(chat_id)
                continue

            if chat_id:
                if chat_id in self._queues:
                    await self._queues[chat_id].put(evt)
                else:
                    self._pending_events.setdefault(chat_id, []).append(evt)

    @staticmethod
    def _parse_event(raw):
        if isinstance(raw, str) and raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"event": "delta", "text": raw}
        if isinstance(raw, bytes):
            try:
                return json.loads(raw.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {"event": "delta", "text": raw.decode(errors="replace")}
        return {"event": "delta", "text": str(raw)}

    # ------------------------------------------------------------------
    # Chat lifecycle
    # ------------------------------------------------------------------

    async def _create_chat(self) -> str:
        await self._ensure_connection()
        self._attach_future = asyncio.Future()
        try:
            await self._ws.send(json.dumps({"type": "new_chat"}))
            chat_id = await asyncio.wait_for(self._attach_future, timeout=15.0)
            logger.info(f"nanobot new chat: {chat_id}")
            return chat_id
        except asyncio.TimeoutError:
            raise RuntimeError("nanobot new_chat timed out")
        finally:
            self._attach_future = None

    def _get_or_create_queue(self, chat_id: str) -> asyncio.Queue:
        if chat_id not in self._queues:
            q = self._queues[chat_id] = asyncio.Queue()
            for evt in self._pending_events.pop(chat_id, []):
                q.put_nowait(evt)
        return self._queues[chat_id]

    # ------------------------------------------------------------------
    # LLMService interface
    # ------------------------------------------------------------------

    async def compose_messages(self, context_id, text, files=None, system_prompt_params=None):
        content = text
        if files:
            parts = [{"type": "text", "text": text}] if text else []
            for f in files:
                if url := f.get("url"):
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            content = parts or text
        return [{"role": "user", "content": content}]

    async def update_context(self, context_id, messages, response_text):
        self.context_tracker.touch(context_id)

    async def get_llm_stream_response(self, context_id, user_id, messages, system_prompt_params=None):
        # Extract user text
        user_content = messages[-1]["content"] if messages else ""
        if isinstance(user_content, list):
            user_text = "\n".join(
                p.get("text", "") for p in user_content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            user_text = str(user_content)

        # Map context_id → chat_id
        chat_id = self._chat_map.get(context_id)
        is_new = chat_id is None
        if is_new:
            chat_id = await self._create_chat()
            self._chat_map[context_id] = chat_id
            self.context_tracker.touch(context_id)

        # Prepend system prompt on first message
        sp = self.system_prompt
        if is_new and sp:
            self._system_prompt_sent[context_id] = True
            effective = f"[System instructions: {sp}]\n\nUser message: {user_text}"
        elif sp and not self._system_prompt_sent.get(context_id):
            self._system_prompt_sent[context_id] = True
            effective = f"[System instructions: {sp}]\n\nUser message: {user_text}"
        else:
            effective = user_text

        if not effective:
            return

        if self.debug:
            logger.info(f"nanobot send (chat={chat_id[:8]}...): {effective[:120]}")

        await self._ensure_connection()
        await self._ws.send(json.dumps({
            "type": "message", "chat_id": chat_id, "content": effective,
        }))

        queue = self._get_or_create_queue(chat_id)

        # stream_end is NOT an end-of-turn signal — nanobot may send
        # multiple stream_end events within a single turn (e.g. after
        # reasoning, after intermediate text before tool execution, and
        # after the final response).  The only reliable terminal signal
        # is turn_end.
        while True:
            evt = await queue.get()
            if evt is None:
                logger.info("[nanobot] ═══ connection closed ═══")
                break

            etype = evt.get("event")
            # Dump full event for comparison
            text_preview = str(evt.get("text", ""))[:80]
            name = evt.get("name", "")
            detail = evt.get("detail", "")
            extra = f" name={name}" if name else ""
            extra += f" detail={detail}" if detail else ""
            # logger.info(f"[nanobot] type={etype} text={text_preview!r}{extra}")

            if etype == "delta":
                yield LLMResponse(context_id=context_id, text=evt.get("text", ""))
            elif etype == "message":
                text = evt.get("text", "")
                if text:
                    yield LLMResponse(context_id=context_id, text=text)
            elif etype == "turn_end":
                logger.info("[nanobot] ═══ turn_end → break ═══")
                break
            elif etype == "error":
                logger.error(f"nanobot error: {evt.get('detail')}")
                break
            elif etype in ("tool_call", "tool_result", "reasoning_delta",
                           "reasoning_end", "stream_end", "session_updated",
                           "goal_status"):
                pass  # intermediate events, logged above
            # catch-all for truly unknown event types
            elif not etype:
                pass
            else:
                logger.info(f"[nanobot] *** unhandled event type: {etype}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self):
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("nanobot closed")
