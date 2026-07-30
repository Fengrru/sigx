"""
QualityGate — filters signals by confidence threshold and deduplication.
"""

import logging
from typing import Dict, List

from ..types import Signal

logger = logging.getLogger(__name__)


class QualityGate:
    """
    Signal quality filter.

    Filters signals by:
    - Minimum confidence threshold
    - Maximum signals per conversation (to prevent over-extraction)
    - Deduplication by (conversation_id, turn_index, signal_type)

    Args:
        min_confidence: Minimum confidence to keep a signal.
        max_per_conversation: Max signals per conversation (0 = unlimited).
        deduplicate: If True, remove duplicate signals for the same turn.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_per_conversation: int = 20,
        deduplicate: bool = True,
    ):
        self.min_confidence = min_confidence
        self.max_per_conversation = max_per_conversation
        self.deduplicate = deduplicate

    def __call__(self, signals: List[Signal]) -> List[Signal]:
        n_total = len(signals)
        filtered = [s for s in signals if s.confidence >= self.min_confidence]
        n_after_conf = len(filtered)
        logger.debug(
            "Confidence filter (>=%.2f): %d → %d",
            self.min_confidence, n_total, n_after_conf,
        )

        if self.deduplicate:
            seen = set()
            unique = []
            for s in filtered:
                key = (s.conversation_id, s.turn_index, s.signal_type)
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            filtered = unique

        if self.max_per_conversation > 0:
            conv_counts: Dict[str, int] = {}
            limited = []
            conv_order: List[str] = []
            for s in filtered:
                cid = s.conversation_id
                if cid not in conv_counts:
                    conv_counts[cid] = 0
                    conv_order.append(cid)
                if conv_counts[cid] < self.max_per_conversation:
                    limited.append(s)
                    conv_counts[cid] += 1
            filtered = limited

        return filtered

    @property
    def n_dropped(self) -> int:
        return getattr(self, "_n_dropped", 0)

    def filter_with_report(self, signals: List[Signal]) -> tuple[List[Signal], Dict]:
        """Filter and return (filtered_signals, report_dict)."""
        n_before = len(signals)
        result = self(signals)
        report = {
            "before": n_before,
            "after": len(result),
            "dropped": n_before - len(result),
            "retention": f"{len(result) / max(n_before, 1) * 100:.1f}%",
            "min_confidence": self.min_confidence,
        }
        return result, report
