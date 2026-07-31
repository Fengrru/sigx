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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ..exceptions import LLMConnectionError, LLMResponseError
from ..types import Signal
from .base import ASSISTANT_ROLES, USER_ROLES, BaseExtractor, format_history

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
        batch_size: Number of turns classified concurrently (thread pool
            size for parallel API calls). 1 = sequential.
        max_context_turns: Max previous turns to include as context.
        extra_headers: Optional dict of extra HTTP headers.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries for a failed API call.
        on_error: "skip" (default) logs failures and drops the turn;
            "raise" raises LLMConnectionError on API failure or
            LLMResponseError when the response cannot be parsed.
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
        timeout: float = 30.0,
        max_retries: int = 2,
        on_error: str = "skip",
    ):
        if on_error not in ("skip", "raise"):
            raise ValueError(f'on_error must be "skip" or "raise", got {on_error!r}')
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
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
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_error = on_error

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

        kwargs: Dict = {"api_key": self.api_key, "timeout": self.timeout}
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
            if role not in USER_ROLES:
                continue

            text = turn.get("content", "").strip()
            if not text or len(text) < 3:
                continue

            if self.require_adjacent_assistant:
                if i == 0 or turns[i - 1].get("role") not in ASSISTANT_ROLES:
                    continue

            candidates.append({"turn_index": i, "text": text})

        if not candidates:
            return []

        # Classify candidates (concurrently when batch_size > 1)
        if self.batch_size > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
                results = list(pool.map(lambda c: self._classify_candidate(turns, c), candidates))
        else:
            results = [self._classify_candidate(turns, cand) for cand in candidates]

        signals: List[Signal] = []
        for cand, result in zip(candidates, results):
            if result is None:
                continue
            turn_idx = cand["turn_index"]
            user_text = cand["text"]

            label, confidence = result
            if label == "neutral" or confidence < self.min_confidence:
                continue

            # Get assistant response before this turn
            assistant_text = ""
            prompt_up_to = turn_idx
            if turn_idx > 0 and turns[turn_idx - 1].get("role") in ASSISTANT_ROLES:
                assistant_text = turns[turn_idx - 1].get("content", "")
                prompt_up_to = turn_idx - 1

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
                        "conversation_prompt": format_history(turns, prompt_up_to),
                    },
                )
            )

        logger.debug("LLMExtractor extracted %d signals from conv %s", len(signals), conv_id)
        return signals

    def _classify_candidate(self, turns: List[Dict], cand: Dict) -> Optional[tuple]:
        """Build history for a candidate turn and classify it."""
        history = self._format_history(turns, cand["turn_index"])
        return self._classify(cand["text"], history)

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
            if role in USER_ROLES:
                lines.append(f"User: {content[:300]}")
            elif role in ASSISTANT_ROLES:
                lines.append(f"Assistant: {content[:300]}")
        return "\n".join(lines) if lines else "(start of conversation)"

    def _classify(self, user_message: str, history: str) -> Optional[tuple]:
        """Call the LLM and parse the JSON response. Returns (label, confidence) or None.

        Raises:
            LLMConnectionError: If all attempts fail and on_error="raise".
            LLMResponseError: If the response is unparseable and on_error="raise".
        """
        client = self._get_client()

        user_prompt = _USER_TEMPLATE.format(history=history, user_message=user_message[:1000])

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
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
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
        else:
            if self.on_error == "raise":
                raise LLMConnectionError(
                    f"LLM API call failed after {self.max_retries + 1} attempts: {last_exc}"
                ) from last_exc
            return None

        raw = response.choices[0].message.content or ""
        parsed = self._parse_response(raw)
        if parsed is None and self.on_error == "raise":
            raise LLMResponseError(f"Could not parse LLM response: {raw[:200]}")
        return parsed

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
