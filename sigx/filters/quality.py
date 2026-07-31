"""
QualityGate — filters signals by confidence threshold and deduplication.

This module provides the QualityGate class for filtering extracted signals
based on confidence thresholds, deduplication rules, and per-conversation
limits to ensure signal quality and prevent over-extraction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from ..exceptions import ConfigurationError, QualityGateError
from ..types import Signal

logger = logging.getLogger(__name__)


class QualityGate:
    """
    Signal quality filter.

    Filters signals by:
    - Minimum confidence threshold
    - Maximum signals per conversation (to prevent over-extraction)
    - Deduplication by (conversation_id, turn_index, signal_type)

    This filter ensures that only high-quality signals are passed through
    for conversion to training data, improving the overall quality of
    the resulting DPO/KTO pairs.

    Args:
        min_confidence: Minimum confidence to keep a signal.
            Signals below this threshold are dropped.
        max_per_conversation: Max signals per conversation (0 = unlimited).
            Prevents any single conversation from dominating the dataset.
        deduplicate: If True, remove duplicate signals for the same turn.
            Ensures each turn produces at most one signal of each type.

    Example:
        >>> gate = QualityGate(min_confidence=0.7, max_per_conversation=10)
        >>> filtered = gate(signals)
        >>> print(f"Filtered {len(signals)} -> {len(filtered)} signals")

        >>> # Get detailed filtering report
        >>> filtered, report = gate.filter_with_report(signals)
        >>> print(f"Retention: {report['retention']}")
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_per_conversation: int = 20,
        deduplicate: bool = True,
    ):
        """
        Initialize the QualityGate.

        Args:
            min_confidence: Minimum confidence threshold in [0.0, 1.0].
            max_per_conversation: Maximum signals per conversation. 0 means no limit.
            deduplicate: Whether to remove duplicate signals.

        Raises:
            ConfigurationError: If parameters are invalid.
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ConfigurationError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")
        if max_per_conversation < 0:
            raise ConfigurationError(
                f"max_per_conversation must be >= 0, got {max_per_conversation}"
            )

        self.min_confidence = min_confidence
        self.max_per_conversation = max_per_conversation
        self.deduplicate = deduplicate
        self._n_dropped = 0

    def __call__(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals based on quality criteria.

        Args:
            signals: List of Signal objects to filter.

        Returns:
            Filtered list of Signal objects.

        Raises:
            QualityGateError: If the input contains objects without the
                Signal interface (confidence/conversation_id attributes).
        """
        n_total = len(signals)
        try:
            filtered = [s for s in signals if s.confidence >= self.min_confidence]
        except (AttributeError, TypeError) as err:
            raise QualityGateError(f"Invalid signal object in input: {err}") from err
        n_after_conf = len(filtered)
        logger.debug(
            "Confidence filter (>=%.2f): %d → %d",
            self.min_confidence,
            n_total,
            n_after_conf,
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

        self._n_dropped = n_total - len(filtered)
        return filtered

    @property
    def n_dropped(self) -> int:
        """
        Get the number of signals dropped in the last filter operation.

        Returns:
            Number of signals dropped (0 if no filtering has run yet).
        """
        return self._n_dropped

    def filter_with_report(self, signals: List[Signal]) -> Tuple[List[Signal], Dict]:
        """
        Filter signals and return a detailed report.

        This method applies the same filtering as __call__ but also
        returns a dictionary with detailed statistics about the
        filtering process.

        Args:
            signals: List of Signal objects to filter.

        Returns:
            A tuple of (filtered_signals, report_dict) where report_dict
            contains:
                - before: Number of signals before filtering
                - after: Number of signals after filtering
                - dropped: Number of signals dropped
                - retention: Percentage of signals retained
                - min_confidence: Minimum confidence threshold used

        Example:
            >>> gate = QualityGate(min_confidence=0.6)
            >>> filtered, report = gate.filter_with_report(signals)
            >>> print(f"Filtered {report['dropped']} signals")
            >>> print(f"Retention rate: {report['retention']}")
        """
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
