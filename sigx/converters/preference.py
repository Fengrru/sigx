"""
Converters — transform extracted signals into training-ready formats.

Supported formats:
- DPO (Direct Preference Optimization): (prompt, chosen, rejected) triples.
  chosen is inferred from subsequent positive turns when available.
- KTO (Kahneman-Tversky Optimization): (prompt, completion, label) triples.
- Rejection: List of (prompt, rejected) pairs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from ..types import KTOExample, PreferencePair, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chosen strategies for DPO conversion
# ---------------------------------------------------------------------------

CHOSEN_NONE = "none"
CHOSEN_SUBSEQUENT = "subsequent"
CHOSEN_LAST_ASSISTANT = "last_assistant"


def _find_chosen_from_signals(
    signal: Signal,
    conversations: Dict[str, List[Dict]],
    positive_turns: Set[Tuple[str, int]],
    negative_turns: Set[Tuple[str, int]],
    strategy: str,
) -> Optional[str]:
    """
    Find a chosen (good) response for a negative signal by looking
    forward in the conversation for positive resolution.

    Strategy 'subsequent':
      1. Find the first positive user turn after this signal's turn.
      2. The assistant response immediately before that positive turn
         is the chosen response (the "corrected" answer).
      3. If no positive turn found, use the last assistant response
         (assuming the conversation eventually resolved).

    Strategy 'last_assistant':
      Use the final assistant response in the conversation.

    Strategy 'none':
      Return None (chosen unknown).
    """
    if strategy == CHOSEN_NONE:
        return None

    turns = conversations.get(signal.conversation_id)
    if not turns or len(turns) <= signal.turn_index + 1:
        return None

    conv_id = signal.conversation_id
    signal_turn = signal.turn_index

    # --- Strategy: subsequent ---
    if strategy == CHOSEN_SUBSEQUENT:
        # Step 1: scan forward for a positive user turn
        for i in range(signal_turn + 1, len(turns)):
            role = turns[i].get("role", "")
            if role in ("user", "human"):
                key = (conv_id, i)
                if key in positive_turns:
                    # Found positive — get the assistant just before it
                    for j in range(i - 1, signal_turn, -1):
                        if turns[j].get("role") in ("assistant", "gpt", "model"):
                            chosen = turns[j].get("content", "").strip()
                            if chosen:
                                logger.debug(
                                    "Found chosen via subsequent positive"
                                    " at turn %d for signal at turn %d",
                                    i,
                                    signal_turn,
                                )
                                return chosen

        # Step 2: no positive — fall back to last assistant if no further complaints
        last_good = None
        for i in range(len(turns) - 1, signal_turn, -1):
            role = turns[i].get("role", "")
            if role in ("user", "human"):
                key = (conv_id, i)
                if key in negative_turns:
                    # User complained again — invalidate anything after this point
                    last_good = None
                    break
            elif role in ("assistant", "gpt", "model"):
                if last_good is None:
                    last_good = turns[i].get("content", "").strip()

        if last_good:
            logger.debug(
                "Found chosen via last_assistant fallback for signal at turn %d",
                signal_turn,
            )
        return last_good

    # --- Strategy: last_assistant ---
    if strategy == CHOSEN_LAST_ASSISTANT:
        for i in range(len(turns) - 1, signal_turn, -1):
            if turns[i].get("role") in ("assistant", "gpt", "model"):
                return turns[i].get("content", "").strip()
        return None

    return None


def _build_prompt(turns: List[Dict], up_to: int) -> str:
    """Build conversation history string up to (but not including) turn `up_to`."""
    parts = []
    for i in range(up_to):
        role = turns[i].get("role", "")
        content = turns[i].get("content", "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            parts.append(f"User: {content}")
        elif role in ("assistant", "gpt", "model"):
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


def _get_assistant_response(turns: List[Dict], before_turn: int) -> Optional[str]:
    """Get the assistant response immediately before the given user turn."""
    for i in range(before_turn - 1, -1, -1):
        if turns[i].get("role") in ("assistant", "gpt", "model"):
            return turns[i].get("content", "").strip()
    return None


def _get_prompt_and_rejected(
    signal: Signal, conversations: Dict[str, List[Dict]]
) -> tuple[Optional[str], Optional[str]]:
    """Get prompt and rejected text for a signal."""
    if signal.conversation_id not in conversations:
        return None, None

    turns = conversations[signal.conversation_id]
    prompt = _build_prompt(turns, signal.turn_index)
    rejected = _get_assistant_response(turns, signal.turn_index)

    return prompt, rejected


def to_dpo(
    signals: List[Signal],
    conversations: Optional[Dict[str, List[Dict]]] = None,
    chosen_strategy: str = CHOSEN_SUBSEQUENT,
) -> List[PreferencePair]:
    """
    Convert signals to DPO preference pairs.

    For each negative signal, creates a (prompt, chosen, rejected) triple.
    chosen is inferred from subsequent positive resolution in the same
    conversation, or the last assistant response (configurable via
    chosen_strategy).

    Args:
        signals: Extracted signals.
        conversations: Map of conversation_id -> turns list.
            If None, uses context stored in signal.context.
        chosen_strategy: How to find the chosen response.
            - "subsequent" (default): find positive turn later in conv,
              use the assistant response before it as chosen.
            - "last_assistant": use the final assistant response.
            - "none": chosen=None (backward compatible).

    Returns:
        List of PreferencePair objects.
    """
    # Build lookup sets for quick turn-type queries
    positive_turns: Set[Tuple[str, int]] = set()
    negative_turns: Set[Tuple[str, int]] = set()
    for sig in signals:
        key = (sig.conversation_id, sig.turn_index)
        if sig.signal_type == "positive":
            positive_turns.add(key)
        elif sig.signal_type in ("negative", "correction", "rephrase", "abandon"):
            negative_turns.add(key)

    pairs: List[PreferencePair] = []

    for sig in signals:
        if sig.signal_type == "positive":
            continue

        if conversations is not None:
            prompt, rejected = _get_prompt_and_rejected(sig, conversations)
        else:
            prompt = sig.context.get("conversation_prompt", "")
            rejected = sig.context.get("rejected_response", "")

        if not rejected:
            continue

        # Infer chosen from subsequent turns
        chosen = None
        if chosen_strategy != CHOSEN_NONE and conversations is not None:
            chosen = _find_chosen_from_signals(
                sig,
                conversations,
                positive_turns,
                negative_turns,
                chosen_strategy,
            )

        pairs.append(
            PreferencePair(
                prompt=prompt or "",
                chosen=chosen[:2000] if chosen else None,
                rejected=rejected[:2000],
                signal_type=sig.signal_type,
                confidence=sig.confidence,
                conversation_id=sig.conversation_id,
            )
        )

    return pairs


def to_kto(
    signals: List[Signal],
    conversations: Optional[Dict[str, List[Dict]]] = None,
) -> List[KTOExample]:
    """
    Convert signals to KTO examples (binary good/bad labels).

    positive -> label=True (desirable)
    rephrase, correction, negative, abandon -> label=False (undesirable)

    Args:
        signals: Extracted signals.
        conversations: Map of conversation_id -> turns list.

    Returns:
        List of KTOExample objects.
    """
    examples: List[KTOExample] = []

    for sig in signals:
        if conversations is not None:
            prompt, rejected = _get_prompt_and_rejected(sig, conversations)
        else:
            prompt = sig.context.get("conversation_prompt", "")
            rejected = sig.context.get("rejected_response", "")

        completion = rejected or sig.context.get("assistant_response", "")
        if not completion:
            continue

        label = sig.signal_type == "positive"

        examples.append(
            KTOExample(
                prompt=prompt or "",
                completion=completion[:2000],
                label=label,
                confidence=sig.confidence,
                conversation_id=sig.conversation_id,
            )
        )

    return examples


def to_rejection(
    signals: List[Signal],
    conversations: Optional[Dict[str, List[Dict]]] = None,
) -> List[Dict]:
    """
    Convert signals to rejection pairs (for rejection sampling).

    Returns a list of dicts with keys:
        prompt, rejected, signal_type, confidence

    Args:
        signals: Extracted signals.
        conversations: Map of conversation_id -> turns list.

    Returns:
        List of dicts.
    """
    pairs = to_dpo(signals, conversations)
    return [
        {
            "prompt": p.prompt,
            "rejected": p.rejected,
            "signal_type": p.signal_type,
            "confidence": p.confidence,
            "conversation_id": p.conversation_id,
        }
        for p in pairs
    ]
