"""Regression tests for bugs fixed after the deep-audit review.

Each test class targets one specific defect so future refactors
cannot silently reintroduce it.
"""

import pytest

from sigx.converters import to_dpo
from sigx.exceptions import ConfigurationError, ConversationFormatError
from sigx.extractors import RephraseDetector, SentimentDetector
from sigx.extractors.rephrase import _scale_confidence
from sigx.filters import QualityGate
from sigx.pipeline import Pipeline
from sigx.types import SIGNAL_TYPES, Signal

REJECTED_TEXT = "Python is a type of snake found in Asia and Africa."

NEGATIVE_CONV = {
    "conversation_id": "leak-1",
    "conversation": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": REJECTED_TEXT},
        {"role": "user", "content": "That's not what I asked"},
    ],
}


def _make_pipeline():
    return Pipeline(
        extractors=[SentimentDetector(min_confidence=0.5)],
        quality_gate=QualityGate(min_confidence=0.5),
    )


class TestPromptLeak:
    """The rejected response must never appear inside its own prompt."""

    def test_dpo_prompt_does_not_contain_rejected(self):
        pairs = _make_pipeline().to_dpo([NEGATIVE_CONV])
        assert len(pairs) >= 1
        for p in pairs:
            assert p.rejected == REJECTED_TEXT[:2000]
            assert p.rejected not in p.prompt

    def test_kto_prompt_does_not_contain_completion(self):
        examples = _make_pipeline().to_kto([NEGATIVE_CONV])
        assert len(examples) >= 1
        for ex in examples:
            assert ex.completion not in ex.prompt

    def test_fallback_context_produces_pairs_without_conversations(self):
        """to_dpo(signals) without a conv map must use signal context."""
        signals = _make_pipeline().run([NEGATIVE_CONV])
        pairs = to_dpo(signals, conversations=None)
        assert len(pairs) >= 1
        assert pairs[0].rejected == REJECTED_TEXT[:500]
        assert pairs[0].rejected not in pairs[0].prompt


class TestMissingConversationId:
    """Conversations without conversation_id must still produce pairs."""

    def test_to_dpo_without_conversation_id(self):
        conv = {"conversation": list(NEGATIVE_CONV["conversation"])}
        pairs = _make_pipeline().to_dpo([conv])
        assert len(pairs) >= 1
        assert pairs[0].conversation_id == "0"

    def test_run_assigns_positional_ids(self):
        convs = [
            {"conversation": list(NEGATIVE_CONV["conversation"])},
            {"conversation": list(NEGATIVE_CONV["conversation"])},
        ]
        signals = _make_pipeline().run(convs)
        assert {s.conversation_id for s in signals} == {"0", "1"}


class TestPositiveGuards:
    """Guards must only suppress positive, not negative/correction."""

    def test_no_thanks_still_yields_negative(self):
        detector = SentimentDetector(min_confidence=0.5)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "Do you have more info?"},
                {"role": "assistant", "content": "Here is more information..."},
                {"role": "user", "content": "No thanks, that's not what I needed."},
            ],
        }
        signals = detector.extract(conv)
        types = [s.signal_type for s in signals]
        assert "positive" not in types
        assert "negative" in types

    def test_not_helpful_yields_negative(self):
        detector = SentimentDetector(min_confidence=0.5)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "Explain recursion"},
                {"role": "assistant", "content": "Recursion is recursion."},
                {"role": "user", "content": "This is not helpful at all."},
            ],
        }
        signals = detector.extract(conv)
        types = [s.signal_type for s in signals]
        assert "positive" not in types
        assert "negative" in types


class TestHighestWeightMatch:
    """The strongest pattern must win, not the first one in the list."""

    def test_strong_negative_beats_weak_positive(self):
        detector = SentimentDetector(min_confidence=0.5)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
                {"role": "user", "content": "Thanks, but that's completely wrong."},
            ],
        }
        signals = detector.extract(conv)
        assert len(signals) == 1
        assert signals[0].signal_type == "negative"
        assert signals[0].confidence == pytest.approx(0.90)


class TestEvaluateFalsePositives:
    """Predictions of types absent from ground truth must count as FP."""

    def test_spurious_type_counts_as_fp(self):
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        benchmark = [
            {
                "conversation_id": "b1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a snake."},
                    {"role": "user", "content": "That's wrong"},
                ],
                # GT labels this turn as rephrase; the pipeline will
                # predict negative — that must be an FP, not ignored.
                "ground_truth": [{"turn_index": 2, "signal_type": "rephrase"}],
            }
        ]
        metrics = pipeline.evaluate(benchmark)
        assert "negative" in metrics["per_type"]
        assert metrics["per_type"]["negative"]["precision"] == 0.0
        assert metrics["overall"]["precision"] == 0.0
        assert metrics["overall"]["recall"] == 0.0

    def test_invalid_benchmark_type_raises(self):
        pipeline = Pipeline(extractors=[SentimentDetector()])
        with pytest.raises(ConfigurationError):
            pipeline.evaluate(123)  # type: ignore[arg-type]


class TestQualityGateDropped:
    def test_n_dropped_reflects_last_run(self):
        gate = QualityGate(min_confidence=0.7)
        signals = [
            Signal(
                conversation_id="1",
                turn_index=i,
                signal_type="negative",
                confidence=c,
                evidence="test",
            )
            for i, c in enumerate([0.5, 0.8, 0.6])
        ]
        filtered = gate(signals)
        assert len(filtered) == 1
        assert gate.n_dropped == 2

    def test_invalid_config_raises(self):
        with pytest.raises(ConfigurationError):
            QualityGate(min_confidence=1.5)


class TestMinTurnLength:
    """Turns shorter than min_turn_length must be skipped, not kept."""

    def test_short_turns_filtered(self):
        detector = RephraseDetector(similarity_threshold=0.3, min_turn_length=50)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python language?"},
                {"role": "assistant", "content": "Python is a language."},
                {"role": "user", "content": "What is Python language??"},
            ],
        }
        # Both user turns are < 50 chars → filtered → no signal.
        assert detector.extract(conv) == []

    def test_long_turns_kept(self):
        detector = RephraseDetector(similarity_threshold=0.3, min_turn_length=20)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python programming language exactly?"},
                {"role": "assistant", "content": "Python is a language."},
                {"role": "user", "content": "What is Python programming language exactly??"},
            ],
        }
        assert len(detector.extract(conv)) == 1


class TestRephraseConfidence:
    """Detected rephrases must survive the default QualityGate (0.6)."""

    def test_confidence_floor_at_threshold(self):
        assert _scale_confidence(0.6, 0.6) == pytest.approx(0.6)
        assert _scale_confidence(1.0, 0.6) == pytest.approx(1.0)

    def test_detected_signal_passes_default_gate(self):
        detector = RephraseDetector(similarity_threshold=0.3)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python programming language?"},
                {"role": "assistant", "content": "Python is a language."},
                {"role": "user", "content": "What is Python programming language and usage?"},
            ],
        }
        signals = detector.extract(conv)
        assert len(signals) == 1
        assert signals[0].confidence >= 0.6

    def test_double_send_without_assistant_is_skipped(self):
        detector = RephraseDetector(similarity_threshold=0.3)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python programming language?"},
                {"role": "user", "content": "What is Python programming language??"},
            ],
        }
        assert detector.extract(conv) == []


class TestCJKRephrase:
    """CJK text must fall back to char n-grams instead of failing."""

    def test_chinese_rephrase_detected(self):
        detector = RephraseDetector(similarity_threshold=0.5, min_turn_length=5)
        conv = {
            "conversation_id": "zh-1",
            "conversation": [
                {"role": "user", "content": "什么是Python编程语言？请介绍它的特点。"},
                {"role": "assistant", "content": "Python是一种蛇。"},
                {"role": "user", "content": "什么是Python编程语言？请再介绍一下它的特点。"},
            ],
        }
        signals = detector.extract(conv)
        assert len(signals) == 1
        assert signals[0].signal_type == "rephrase"


class TestPipelineStreaming:
    """Pipeline must consume generators in a single pass."""

    def test_run_accepts_generator(self):
        gen = (dict(NEGATIVE_CONV, conversation_id=str(i)) for i in range(5))
        signals = _make_pipeline().run(gen)
        assert len(signals) == 5

    def test_to_dpo_accepts_generator(self):
        gen = (dict(NEGATIVE_CONV, conversation_id=str(i)) for i in range(3))
        pairs = _make_pipeline().to_dpo(gen)
        assert len(pairs) == 3

    def test_non_dict_conversation_raises(self):
        with pytest.raises(ConversationFormatError):
            _make_pipeline().run(["not a dict"])


class TestSignalTypesConstant:
    def test_signal_types_exported(self):
        assert SIGNAL_TYPES == {"rephrase", "correction", "positive", "negative", "abandon"}
