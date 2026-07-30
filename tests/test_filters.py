"""Tests for filters module."""

from sigx.filters import QualityGate
from sigx.types import Signal


class TestQualityGate:
    def test_empty_signals(self):
        gate = QualityGate()
        signals = gate([])
        assert signals == []

    def test_low_confidence_filtered(self):
        gate = QualityGate(min_confidence=0.7)
        signals = [
            Signal(
                conversation_id="1",
                turn_index=0,
                signal_type="negative",
                confidence=0.5,
                evidence="test",
            ),
            Signal(
                conversation_id="1",
                turn_index=1,
                signal_type="negative",
                confidence=0.8,
                evidence="test",
            ),
        ]
        filtered = gate(signals)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.8

    def test_deduplication(self):
        gate = QualityGate(deduplicate=True)
        signals = [
            Signal(
                conversation_id="1",
                turn_index=0,
                signal_type="negative",
                confidence=0.8,
                evidence="test",
            ),
            Signal(
                conversation_id="1",
                turn_index=0,
                signal_type="negative",
                confidence=0.9,
                evidence="test",
            ),
        ]
        filtered = gate(signals)
        assert len(filtered) == 1

    def test_max_per_conversation(self):
        gate = QualityGate(max_per_conversation=2)
        signals = [
            Signal(
                conversation_id="1",
                turn_index=i,
                signal_type="negative",
                confidence=0.8,
                evidence="test",
            )
            for i in range(5)
        ]
        filtered = gate(signals)
        assert len(filtered) == 2

    def test_filter_with_report(self):
        gate = QualityGate(min_confidence=0.7)
        signals = [
            Signal(
                conversation_id="1",
                turn_index=0,
                signal_type="negative",
                confidence=0.5,
                evidence="test",
            ),
            Signal(
                conversation_id="1",
                turn_index=1,
                signal_type="negative",
                confidence=0.8,
                evidence="test",
            ),
        ]
        filtered, report = gate.filter_with_report(signals)
        assert len(filtered) == 1
        assert report["before"] == 2
        assert report["after"] == 1
        assert report["dropped"] == 1
