"""LLM service abstractions and a direct OpenAI-compatible streaming client."""

import asyncio
import inspect
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Dict, List, Optional

import httpx

from ..context_manager import ContextTracker
from ..models import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

# Emoji ranges that would be vocalized by TTS as their names (e.g. 🐈 → "猫",
# 🇨🇳 → flag name). Stripped before synthesis. Includes regional indicators
# (flags), emoticons, pictographs, dingbats, and the FE0F variation selector.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F700-\U0001F77F"  # Alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric shapes extended
    "\U0001F800-\U0001F8FF"  # Supplemental arrows
    "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # Chess symbols
    "\U0001FA70-\U0001FAFF"  # Symbols & pictographs extended-A
    "\U0001F1E6-\U0001F1FF"  # Regional indicators (flags)
    "\U00002702-\U000027B0"  # Dingbats
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002B00-\U00002BFF"  # Misc symbols and arrows
    "\U0000FE0F"             # Variation selector-16
    "]+"
)


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
        #: 首个 token 到达时回调（用来区分"模型开口慢"与"第一个句子边界来得晚"）
        self.on_first_token: Optional[Callable[[], None]] = None
        #: 真正发出 HTTP 请求时回调（用来区分"请求前的本地开销"与"网络+模型耗时"）
        self.on_request_start: Optional[Callable[[], None]] = None

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

    def add_tool(self, name: str, spec: Dict, func: Callable, instruction: str = None) -> Tool:
        """注册一个可供模型调用的工具。

        *spec* 是 OpenAI 工具描述(``{"type":"function","function":{...}}``),
        会随每次请求一起发给模型; *func* 是异步或同步实现, 通过
        :meth:`execute_tool` 调用。
        """
        tool = Tool(name=name, spec=spec, func=func, instruction=instruction)
        self.tools[name] = tool
        return tool

    async def execute_tool(self, name: str, arguments: dict, metadata: dict = None):
        tool = self.tools[name]
        if "metadata" in inspect.signature(tool.func).parameters:
            arguments["metadata"] = metadata
        result = tool.func(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def append_turn(self, context_id: str, user_text: str, assistant_text: str):
        """把一轮对话(用户输入 + 助手回复)落进历史。

        多轮工具调用时, ``chat_stream(persist=False)`` 不会自己落历史
        (否则每轮都会写一条, 且会把 tool 管道消息带进历史), 由上层在本轮真正
        结束时调一次这里, 保证历史里始终是干净的 ``user`` / ``assistant`` 成对消息。
        """
        if not user_text and not assistant_text:
            return
        messages = [{"role": "user", "content": user_text}] if user_text else []
        await self.update_context(context_id, messages, assistant_text)

    # -- lifecycle ---------------------------------------------------------

    async def warmup(self):
        """启动预热钩子(建连接/SSL 上下文等)。默认不做任何事。"""

    async def shutdown(self):
        """释放资源。默认不做任何事。"""

    # -- main streaming entry point ---------------------------------------

    async def chat_stream(
        self,
        context_id: str,
        user_id: str,
        text: str,
        files: List[Dict] = None,
        system_prompt_params: Dict = None,
        extra_messages: List[Dict] = None,
        persist: bool = True,
    ) -> AsyncGenerator[LLMResponse, None]:
        """High-level streaming entry point with text splitting & voice extraction.

        Parameters
        ----------
        extra_messages:
            只在本轮请求里追加的消息(工具轮次: ``assistant.tool_calls`` +
            ``role:"tool"`` 结果)。不写进历史。
        persist:
            False 表示本轮结束不落历史 —— 多轮工具调用时由上层在本轮真正结束时
            调 :meth:`append_turn` 统一落一次。
        """
        logger.info(f"User: {text}")
        text = self._request_filter(text)

        if not text and not files and not extra_messages:
            return

        extra_messages = extra_messages or []
        messages = await self.compose_messages(context_id, text, files, system_prompt_params,
                                              extra_messages)
        user_msg_index = len(messages) - 1 - len(extra_messages)

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
                # 工具调用前若还有没吐完的文本(如确认话术), 必须先切出去 ——
                # 否则这句话会一直压在 buffer 里, 直到工具跑完才出声。
                if stream_buffer.strip():
                    voice = extract_voice(stream_buffer)
                    yield LLMResponse(context_id, stream_buffer, voice)
                    response_text += stream_buffer
                    stream_buffer = ""
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
        if persist and 0 <= user_msg_index < len(messages):
            await self.update_context(context_id, [messages[user_msg_index]], response_text)

    # -- helpers ----------------------------------------------------------

    def _replace_last_opt_split(self, text: str) -> str:
        return self._opt_split_re.sub(r"\1|", text)

    @staticmethod
    def _remove_control_tags(text: str) -> str:
        text = re.sub(r"\[(\w+):([^\]]+)\]", "", text)
        return LLMService._clean_for_tts(text)

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """Strip markdown / emoji / list formatting so TTS won't read it aloud.

        Only markers that would be vocalized as garbage (bold asterisks, list
        dashes, emoji names, URLs) are removed — the spoken content itself is
        preserved. E.g. ``**多云**🌦️`` → ``多云``. Sentence boundaries (``。``,
        line breaks) are kept so TTS still gets natural pauses.
        """
        if not text:
            return text

        # Inline markdown emphasis / code / links → keep inner content
        t = re.sub(r"\*\*(.+?)\*\*", r"\1", text)          # **bold**
        t = re.sub(r"\*{2,}", "", t)                       # stray **
        t = re.sub(r"`([^`]+)`", r"\1", t)                 # `code`
        t = re.sub(r"~~(.+?)~~", r"\1", t)                 # ~~strike~~
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)     # [text](url)

        # Line-level formatting markers
        t = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]*", "", t, flags=re.M)   # headings
        t = re.sub(r"^[ \t]*[-*+][ \t]+", "", t, flags=re.M)        # bullets
        t = re.sub(r"^[ \t]*\d{1,3}[.、)][ \t]+", "", t, flags=re.M)  # numbered lists
        t = re.sub(r"^[ \t]*>[ \t]?", "", t, flags=re.M)            # blockquotes
        t = t.replace("|", " ")                                     # tables

        # Emojis & symbols TTS would vocalize as names
        t = EMOJI_PATTERN.sub("", t)
        # URLs are not meant to be read aloud
        t = re.sub(r"https?://\S+", "", t)

        # Normalize whitespace: collapse spaces, cap blank lines at two
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


# ========================================================================
# Direct OpenAI-compatible LLM service (replaces the nanobot gateway)
# ========================================================================

class DeepSeekLLMService(LLMService):
    """直连 OpenAI 兼容 ``/chat/completions`` 的流式 LLM 服务。

    与原来的 nanobot 网关版相比，最大的区别是**对话历史在本地维护**
    （nanobot 是服务端维护）：``compose_messages`` 把 system + 历史 + 本轮用户
    输入一起发出去，``update_context`` 负责把用户消息和助手回复追加进历史。

    Parameters
    ----------
    base_url:
        OpenAI 兼容端点，默认 ``https://api.deepseek.com``。
    api_key:
        API key，默认取 ``DEEPSEEK_API_KEY`` 环境变量。
    model:
        模型名，默认 ``deepseek-flash``。
    thinking:
        是否开启思考模式，默认 False。语音场景首字优先，会显式发送
        ``thinking={"type": "disabled"}`` 让服务端跳过思维链。
    history_limit:
        每个 context 最多保留的历史消息条数（超出丢最早的）。
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.deepseek.com",
        api_key: str = None,
        model: str = "deepseek-flash",
        system_prompt: str = None,
        temperature: float = 0.5,
        max_tokens: int = 300,
        thinking: bool = False,
        history_limit: int = 20,
        timeout: float = 60.0,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        context_tracker: ContextTracker = None,
        debug: bool = False,
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            voice_text_tag=voice_text_tag,
            context_tracker=context_tracker or ContextTracker(),
            debug=debug,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LLM API key is required. Set DEEPSEEK_API_KEY env var or pass "
                "llm_api_key=... (否则请求会带着 'Bearer None' 拿到 401)"
            )
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.history_limit = history_limit
        self.timeout = timeout
        self._history: Dict[str, List[Dict]] = {}
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warmup(self):
        """提前把 SSL 上下文和 httpx client 建好（启动时调一次）。

        ``httpx.AsyncClient()`` 会同步创建 ssl 上下文（本机约 1s/个，而且是在
        事件循环线程上执行），如果留到第一轮对话里建，这几秒就直接变成首字延时。
        这里用 ``asyncio.to_thread`` 在线程里建，不占用事件循环。
        """
        if self._client is None:
            verify = await asyncio.to_thread(self._build_ssl_context)
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                verify=verify,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

    @staticmethod
    def _build_ssl_context():
        import ssl

        import certifi

        return ssl.create_default_context(cafile=certifi.where())

    async def shutdown(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # LLMService interface
    # ------------------------------------------------------------------

    async def compose_messages(self, context_id, text, files=None, system_prompt_params=None,
                               extra_messages=None):
        content = text
        if files:
            parts = [{"type": "text", "text": text}] if text else []
            for f in files:
                if url := f.get("url"):
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            content = parts or text

        messages: List[Dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history.get(context_id, []))
        if text or files:
            messages.append({"role": "user", "content": content})
        if extra_messages:
            messages.extend(extra_messages)
        return messages

    async def update_context(self, context_id, messages, response_text):
        history = self._history.setdefault(context_id, [])
        history.extend(messages)
        if response_text:
            history.append({"role": "assistant", "content": response_text})
        if len(history) > self.history_limit:
            del history[:-self.history_limit]
        self.context_tracker.touch(context_id)

    async def get_llm_stream_response(self, context_id, user_id, messages,
                                      system_prompt_params=None):
        await self.warmup()

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.tools:
            payload["tools"] = [t.spec for t in self.tools.values()]
        if not self.thinking:
            payload["thinking"] = {"type": "disabled"}  # 跳过思维链, 首字优先
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # tool_calls 在 SSE 里是**分片**下发的: index 索引, function.name 只出现在
        # 首片, arguments 逐片拼接, 结束信号是 finish_reason="tool_calls"。
        # 所以必须累加, 不能当普通 content 处理。
        pending_calls: Dict[int, Dict[str, str]] = {}
        first_token_reported = False

        if self.on_request_start:
            self.on_request_start()

        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                logger.error(f"LLM HTTP {resp.status_code}: {body[:300]}")
                raise RuntimeError(f"LLM HTTP {resp.status_code}")

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    choice = json.loads(data)["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    if not first_token_reported and self.on_first_token:
                        first_token_reported = True
                        self.on_first_token()
                    yield LLMResponse(context_id=context_id, text=delta["content"])
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = pending_calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

        for idx in sorted(pending_calls):
            slot = pending_calls[idx]
            if not slot["name"]:
                continue
            raw = slot["args"].strip()
            if not raw:
                args = {}
            else:
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"工具 {slot['name']} 参数不是合法 JSON: {raw[:120]}")
                    args = {"_raw": raw}
            yield LLMResponse(
                context_id=context_id,
                tool_call=ToolCall(id=slot["id"] or None, name=slot["name"], arguments=args),
            )
