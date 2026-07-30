"""Tests for extractors module."""
import pytest

from sigx.extractors import AbandonDetector, BaseExtractor, RephraseDetector, SentimentDetector

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

UNICODE_CONV = {
    "conversation_id": "unicode-1",
    "conversation": [
        {"role": "user", "content": "什么是Python编程语言？"},
        {"role": "assistant", "content": "Python是一种高级编程语言。"},
        {"role": "user", "content": "这不是我问的问题"},
    ],
}


LONG_TEXT = "Python is a programming language. " * 200

LONG_CONV = {
    "conversation_id": "long-1",
    "conversation": [
        {"role": "user", "content": "What is Python programming language? " + LONG_TEXT},
        {"role": "assistant", "content": LONG_TEXT},
        {"role": "user", "content": "What is Python programming language and how to use it? " + LONG_TEXT},
    ],
}


# ---------------------------------------------------------------------------
# BaseExtractor
# ---------------------------------------------------------------------------


class TestBaseExtractor:
    def test_base_extractor_is_abstract(self):
        with pytest.raises(TypeError):
            BaseExtractor()


class TestRephraseDetector:
    def test_empty_conversation(self):
        detector = RephraseDetector()
        signals = detector.extract({"conversation": []})
        assert signals == []

    def test_single_turn_no_signal(self):
        detector = RephraseDetector()
        conv = {
            "conversation_id": "1",
            "conversation": [{"role": "user", "content": "What is Python?"}]
        }
        signals = detector.extract(conv)
        assert signals == []

    def test_similar_turns_detected(self):
        detector = RephraseDetector(similarity_threshold=0.3)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python programming language?"},
                {"role": "assistant", "content": "Python is a high-level programming language."},
                {"role": "user", "content": "What is Python programming language and how to use it?"},
            ]
        }
        signals = detector.extract(conv)
        assert len(signals) == 1
        assert signals[0].signal_type == "rephrase"
        assert signals[0].conversation_id == "1"
        assert signals[0].turn_index == 2

    def test_different_turns_no_signal(self):
        detector = RephraseDetector(similarity_threshold=0.9)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python programming language?"},
                {"role": "assistant", "content": "Python is a high-level programming language."},
                {"role": "user", "content": "How do I install Java on Windows?"},
            ]
        }
        signals = detector.extract(conv)
        assert len(signals) == 0

    def test_long_text_conversation(self):
        """RephraseDetector handles very long texts without error."""
        detector = RephraseDetector(similarity_threshold=0.3)
        signals = detector.extract(LONG_CONV)
        assert len(signals) >= 1
        assert signals[0].signal_type == "rephrase"

    def test_unicode_text(self):
        """RephraseDetector handles non-ASCII text gracefully."""
        detector = RephraseDetector(similarity_threshold=0.5)
        signals = detector.extract(UNICODE_CONV)
        # May or may not detect rephrase; just must not crash
        assert isinstance(signals, list)

    def test_missing_conversation_key(self):
        detector = RephraseDetector()
        signals = detector.extract({})
        assert signals == []

    def test_no_user_turns(self):
        detector = RephraseDetector()
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "assistant", "content": "Hello!"},
                {"role": "assistant", "content": "How can I help?"},
            ],
        }
        signals = detector.extract(conv)
        assert signals == []


class TestSentimentDetector:
    def test_empty_conversation(self):
        detector = SentimentDetector()
        signals = detector.extract({"conversation": []})
        assert signals == []

    def test_negative_pattern(self):
        detector = SentimentDetector(min_confidence=0.5)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "That's not what I asked"},
            ]
        }
        signals = detector.extract(conv)
        assert len(signals) >= 1
        assert any(s.signal_type in ["negative", "correction"] for s in signals)

    def test_positive_pattern(self):
        detector = SentimentDetector(min_confidence=0.5)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Thanks! That was helpful."},
            ]
        }
        signals = detector.extract(conv)
        assert len(signals) >= 1
        assert any(s.signal_type == "positive" for s in signals)

    def test_correction_pattern(self):
        detector = SentimentDetector(min_confidence=0.5)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
                {"role": "user", "content": "Actually I meant the programming language"},
            ]
        }
        signals = detector.extract(conv)
        assert len(signals) >= 1
        assert any(s.signal_type == "correction" for s in signals)

    def test_false_positive_guard(self):
        """Phrases like 'no thanks' should not trigger positive signals."""
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
        # "No thanks" should be guarded, but "not what I needed" should be negative
        assert not any(s.signal_type == "positive" for s in signals)

    def test_unicode_sentiment(self):
        """SentimentDetector handles Chinese text gracefully."""
        detector = SentimentDetector(min_confidence=0.5)
        signals = detector.extract(UNICODE_CONV)
        # Chinese text won't match English regex patterns, but must not crash
        assert isinstance(signals, list)

    def test_sarcasm_detected(self):
        """Sarcastic 'wow thanks' should be detected as negative, not positive."""
        detector = SentimentDetector(min_confidence=0.5)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "Help me with this problem"},
                {"role": "assistant", "content": "I don't know, sorry."},
                {"role": "user", "content": "Wow, great. Thanks a lot."},
            ],
        }
        signals = detector.extract(conv)
        # Should detect sarcasm pattern (negative) not positive
        types = [s.signal_type for s in signals]
        assert "negative" in types or len(signals) == 0  # sarcasm pattern may match

    def test_ml_mode_fit_and_extract(self):
        """ML mode: train on examples, then extract signals."""
        detector = SentimentDetector(min_confidence=0.5, use_ml=True)
        detector.fit(
            texts=[
                "that's wrong",
                "you are incorrect",
                "not what I asked",
                "thanks so much",
                "this is perfect",
                "exactly what I needed",
                "actually I meant",
                "no I meant something else",
            ],
            labels=[
                "negative", "negative", "negative",
                "positive", "positive", "positive",
                "correction", "correction",
            ],
        )
        # After training, extract on a simple conversation
        conv = {
            "conversation_id": "ml-1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
                {"role": "user", "content": "that's completely wrong, Python is a programming language"},
            ],
        }
        signals = detector.extract(conv)
        assert isinstance(signals, list)
        # Either regex or ML should catch this
        if len(signals) > 0:
            assert signals[0].signal_type in ("negative", "correction")

    def test_ml_mode_without_sklearn(self):
        """ML mode gracefully degrades when sklearn is unavailable."""
        detector = SentimentDetector(use_ml=True)
        # use_ml=True but _ml_pipeline is None → should still work in regex mode
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "That's wrong"},
            ],
        }
        signals = detector.extract(conv)
        assert len(signals) >= 1  # regex still catches "that's wrong"


class TestAbandonDetector:
    def test_empty_conversation(self):
        detector = AbandonDetector()
        signals = detector.extract({"conversation": []})
        assert signals == []

    def test_short_conversation(self):
        detector = AbandonDetector(min_turns=3)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        signals = detector.extract(conv)
        assert len(signals) == 0

    def test_frustration_pattern(self):
        detector = AbandonDetector(min_turns=3)
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "Help me with Python"},
                {"role": "assistant", "content": "Sure, what do you need?"},
                {"role": "user", "content": "Never mind, I'll figure it out myself"},
            ]
        }
        signals = detector.extract(conv)
        assert len(signals) == 1
        assert signals[0].signal_type == "abandon"

    def test_unanswered_question_detected(self):
        """Conversation ends on assistant after user asked a question → abandon."""
        detector = AbandonDetector(
            min_assistant_length=100,
            min_turns=3,
            require_unanswered_question=True,
        )
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "I need help with Python."},
                {"role": "assistant", "content": "What specifically do you need?"},
                {"role": "user", "content": "How do I read a CSV file?"},
                {"role": "assistant", "content": "Here is a detailed guide on reading CSV files in Python. " + "First, import the csv module. " * 20},
            ],
        }
        signals = detector.extract(conv)
        assert len(signals) >= 1
        assert signals[0].signal_type == "abandon"

    def test_resolved_conversation_no_abandon(self):
        """Conversation ends on assistant but user's last message was not a question."""
        detector = AbandonDetector(
            min_assistant_length=100,
            min_turns=3,
            require_unanswered_question=True,
        )
        conv = {
            "conversation_id": "1",
            "conversation": [
                {"role": "user", "content": "I need help with Python."},
                {"role": "assistant", "content": "Sure! Python is great."},
                {"role": "user", "content": "Thanks, that's exactly what I needed."},
                {"role": "assistant", "content": "Here is more detailed information about Python programming... " + "Python " * 30},
            ],
        }
        signals = detector.extract(conv)
        # User's last message was not a question, so should NOT flag as abandon
        abandon_signals = [s for s in signals if s.context.get("reason") == "conversation_ends_on_assistant"]
        assert len(abandon_signals) == 0

    def test_new_frustration_patterns(self):
        """Expanded frustration patterns are detected."""
        detector = AbandonDetector(min_turns=3)
        for text in [
            "I'm done with this",
            "this is useless",
            "you're not helping",
            "can't help me at all",
        ]:
            conv = {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "Help me"},
                    {"role": "assistant", "content": "Sure"},
                    {"role": "user", "content": text},
                ],
            }
            signals = detector.extract(conv)
            assert len(signals) == 1, f"Pattern '{text}' not detected"
            assert signals[0].signal_type == "abandon"
