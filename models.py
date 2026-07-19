"""Core data models for the voice assistant pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Any = None


@dataclass
class LLMResponse:
    """A single chunk from the LLM stream."""
    context_id: str
    text: Optional[str] = None
    voice_text: Optional[str] = None
    tool_call: Optional[ToolCall] = None


@dataclass
class STSRequest:
    """Input to the speech-to-speech pipeline."""
    type: str = "start"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    audio_duration: float = 0
    files: Optional[List[Dict[str, str]]] = None
    system_prompt_params: Optional[Dict[str, Any]] = None


@dataclass
class STSResponse:
    """Output from the speech-to-speech pipeline."""
    type: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    text: Optional[str] = None
    voice_text: Optional[str] = None
    audio_data: Optional[bytes] = None
    tool_call: Optional[ToolCall] = None
    metadata: Optional[dict] = None
