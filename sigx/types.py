"""
Core data types for the SigX library.

This module defines the fundamental data structures used throughout SigX:
- Signal: Represents an extracted implicit feedback signal
- PreferencePair: A DPO training pair (prompt, chosen, rejected)
- KTOExample: A KTO training example (prompt, completion, label)
- SignalReport: Summary statistics for extracted signals
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    """
    A signal extracted from a turn in a conversation.

    Attributes:
        conversation_id: Unique identifier for the conversation.
        turn_index: Which user turn (0-indexed) triggered this signal.
        signal_type: One of "rephrase", "correction", "positive", "negative", "abandon".
        confidence: Extractor confidence in [0.0, 1.0].
        evidence: The user text that triggered this signal.
        context: Additional metadata (similarity scores, matched patterns, etc.).
    """

    conversation_id: str
    turn_index: int
    signal_type: str
    confidence: float
    evidence: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair:
    """
    A pair for DPO training derived from implicit signals.

    Attributes:
        prompt: The conversation history leading up to the model response.
        chosen: The preferred response (may be None if unknown).
        rejected: The model's response that the user implicitly rejected.
        signal_type: How this pair was derived.
        confidence: Signal confidence in [0.0, 1.0].
        conversation_id: Source conversation ID for traceability.
    """

    prompt: str
    chosen: Optional[str]
    rejected: str
    signal_type: str
    confidence: float
    conversation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KTOExample:
    """
    An example for KTO training (binary good/bad label).

    Attributes:
        prompt: The conversation history.
        completion: The model's response.
        label: True for desirable, False for undesirable.
        confidence: Signal confidence in [0.0, 1.0].
        conversation_id: Source conversation ID.
    """

    prompt: str
    completion: str
    label: bool
    confidence: float
    conversation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalReport:
    """
    Summary report of extracted signals from a batch of conversations.
    """

    total_conversations: int
    total_turns: int
    total_signals: int
    by_type: Dict[str, int]
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    conversations_with_signals: int
    signals_per_conversation: float

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  SigX Signal Extraction Report",
            "=" * 50,
            f"  Conversations:     {self.total_conversations}",
            f"  Total turns:       {self.total_turns}",
            f"  Signals extracted: {self.total_signals}",
            f"  By type:           {self.by_type}",
            f"  Mean confidence:   {self.mean_confidence:.3f}",
            f"  Confidence range:  [{self.min_confidence:.3f}, {self.max_confidence:.3f}]",
            f"  Signal density:    {self.signals_per_conversation:.2f}/conv",
            "=" * 50,
        ]
        return "\n".join(lines)


def generate_report(signals: List[Signal], n_conversations: int, n_turns: int) -> SignalReport:
    """Generate a summary report from a list of signals."""
    if not signals:
        return SignalReport(
            total_conversations=n_conversations,
            total_turns=n_turns,
            total_signals=0,
            by_type={},
            mean_confidence=0.0,
            min_confidence=0.0,
            max_confidence=0.0,
            conversations_with_signals=0,
            signals_per_conversation=0.0,
        )

    by_type: Dict[str, int] = {}
    confidences = [s.confidence for s in signals]
    seen_convos = set(s.conversation_id for s in signals)

    for s in signals:
        by_type[s.signal_type] = by_type.get(s.signal_type, 0) + 1

    return SignalReport(
        total_conversations=n_conversations,
        total_turns=n_turns,
        total_signals=len(signals),
        by_type=by_type,
        mean_confidence=sum(confidences) / len(confidences),
        min_confidence=min(confidences),
        max_confidence=max(confidences),
        conversations_with_signals=len(seen_convos),
        signals_per_conversation=len(signals) / max(n_conversations, 1),
    )


def save_signals_jsonl(signals: List[Signal], path: str) -> None:
    """Save signals to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for sig in signals:
            f.write(json.dumps(asdict(sig), ensure_ascii=False) + "\n")


def save_pairs_jsonl(pairs: List[PreferencePair], path: str) -> None:
    """Save preference pairs to JSONL (compatible with TRL)."""
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
