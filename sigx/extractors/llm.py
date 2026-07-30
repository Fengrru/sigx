"""
LLMExtractor — LLM-based signal classification for high-accuracy extraction.

Uses any OpenAI-compatible API (OpenAI, vLLM, Ollama, LM Studio, etc.)
to classify user turns as positive / negative / correction / neutral.

This extractor provides significantly higher accuracy than regex-only
approaches, at the cost of API latency and cost.  Recommended for
production pipelines where signal quality matters.

Requires: pip install openai
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from ..types import Signal
from .base import BaseExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a conversation analyst. Classify the user's last message as one of:

- "positive"   — user is satisfied, grateful, or confirms the answer is good
- "negative"   — user is dissatisfied, says the answer is wrong or unhelpful
- "correction" — user corrects or clarifies what they actually meant
- "neutral"    — none of the above (follow-up question, chitchat, etc.)

Reply ONLY with a JSON object: {"label": "<type>", "confidence": <0.0-1.0>}\
"""

_USER_TEMPLATE = """\
Conversation so far:
{history}

User's latest message: "{user_message}"

Classification:\
"""

# ---------------------------------------------------------------------------
# LLMExtractor
# ---------------------------------------------------------------------------


class LLMExtractor(BaseExtractor):
    """
    Classify user feedback using an LLM via OpenAI-compatible API.

    Supports OpenAI, Azure OpenAI, vLLM, Ollama, LM Studio, and any
    other provider that exposes an OpenAI-compatible /v1/chat/completions
    endpoint.

    Args:
        model: Model name (e.g. "gpt-4o-mini", "qwen2.5-7b-instruct").
        base_url: API base URL. Defaults to OpenAI.
            For Ollama: "http://localhost:11434/v1"
            For vLLM:   "http://localhost:8000/v1"
        api_key: API key. Defaults to OPENAI_API_KEY env var.
        min_confidence: Minimum confidence to emit a signal.
        require_adjacent_assistant: Only classify turns that follow
            an assistant message.
        max_tokens: Max tokens for the LLM response.
        temperature: LLM temperature (0 = deterministic).
        batch_size: Max turns to send in one API call (batching is
            done by sending multiple user messages in sequence; the
            LLM processes them one at a time in the prompt).
        max_context_turns: Max previous turns to include as context.
        extra_headers: Optional dict of extra HTTP headers.
    """

    name = "llm"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        min_confidence: float = 0.6,
        require_adjacent_assistant: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
        batch_size: int = 1,
        max_context_turns: int = 6,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.min_confidence = min_confidence
        self.require_adjacent_assistant = require_adjacent_assistant
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_context_turns = max_context_turns
        self.extra_headers = extra_headers

        self._client: Any = None
        self._checked_import = False

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self._checked_import:
            try:
                from openai import OpenAI
            except ImportError as err:
                raise ImportError(
                    "LLMExtractor requires the 'openai' package. Install with: pip install openai"
                ) from err
            self._checked_import = True

        from openai import OpenAI

        kwargs: Dict = {"api_key": self.api_key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.extra_headers is not None:
            kwargs["default_headers"] = self.extra_headers

        self._client = OpenAI(**kwargs)
        logger.info(
            "LLMExtractor client created: model=%s base_url=%s",
            self.model,
            self.base_url or "default",
        )
        return self._client

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, conversation: Dict) -> List[Signal]:
        turns = conversation.get("conversation", [])
        conv_id = conversation.get("conversation_id", "")

        if not turns:
            return []

        # Collect candidate user turns
        candidates: List[Dict] = []
        for i, turn in enumerate(turns):
            role = turn.get("role", "")
            if role not in ("user", "human"):
                continue

            text = turn.get("content", "").strip()
            if not text or len(text) < 3:
                continue

            if self.require_adjacent_assistant:
                if i == 0 or turns[i - 1].get("role") not in ("assistant", "gpt", "model"):
                    continue

            candidates.append({"turn_index": i, "text": text})

        if not candidates:
            return []

        # Build context windows and classify each candidate
        signals: List[Signal] = []
        for cand in candidates:
            turn_idx = cand["turn_index"]
            user_text = cand["text"]

            # Build conversation history up to this turn
            history = self._format_history(turns, turn_idx)

            # Get assistant response before this turn
            assistant_text = ""
            if turn_idx > 0 and turns[turn_idx - 1].get("role") in ("assistant", "gpt", "model"):
                assistant_text = turns[turn_idx - 1].get("content", "")

            # Classify via LLM
            result = self._classify(user_text, history)
            if result is None:
                continue

            label, confidence = result
            if label == "neutral" or confidence < self.min_confidence:
                continue

            signals.append(
                Signal(
                    conversation_id=conv_id,
                    turn_index=turn_idx,
                    signal_type=label,
                    confidence=confidence,
                    evidence=user_text[:500],
                    context={
                        "method": "llm",
                        "model": self.model,
                        "assistant_response": assistant_text[:500],
                    },
                )
            )

        logger.debug("LLMExtractor extracted %d signals from conv %s", len(signals), conv_id)
        return signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_history(self, turns: List[Dict], up_to: int) -> str:
        """Format the conversation history up to (but not including) turn `up_to`."""
        # Only include the last N turns for context
        start = max(0, up_to - self.max_context_turns)
        lines: List[str] = []
        for i in range(start, up_to):
            role = turns[i].get("role", "")
            content = turns[i].get("content", "").strip()
            if not content:
                continue
            if role in ("user", "human"):
                lines.append(f"User: {content[:300]}")
            elif role in ("assistant", "gpt", "model"):
                lines.append(f"Assistant: {content[:300]}")
        return "\n".join(lines) if lines else "(start of conversation)"

    def _classify(self, user_message: str, history: str) -> Optional[tuple]:
        """Call the LLM and parse the JSON response. Returns (label, confidence) or None."""
        client = self._get_client()

        user_prompt = _USER_TEMPLATE.format(history=history, user_message=user_message[:1000])

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            logger.warning("LLM API call failed: %s", exc)
            return None

        raw = response.choices[0].message.content or ""
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> Optional[tuple]:
        """Parse the LLM JSON output into (label, confidence)."""
        # Try direct JSON parse
        for candidate in (_extract_json_object(raw), raw):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                label = str(data.get("label", "")).strip().lower()
                confidence = float(data.get("confidence", 0.0))
                if label in ("positive", "negative", "correction", "neutral"):
                    return label, round(confidence, 4)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        logger.debug("Failed to parse LLM response: %s", raw[:200])
        return None


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object {...} from a string."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
