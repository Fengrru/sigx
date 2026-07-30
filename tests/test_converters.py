"""Tests for converters module."""
from sigx.converters import to_dpo, to_kto, to_rejection
from sigx.types import Signal


class TestToDpo:
    def test_empty_signals(self):
        pairs = to_dpo([])
        assert pairs == []

    def test_positive_signal_excluded(self):
        signals = [
            Signal(
                conversation_id="1",
                turn_index=2,
                signal_type="positive",
                confidence=0.8,
                evidence="thanks"
            )
        ]
        pairs = to_dpo(signals)
        assert len(pairs) == 0

    def test_negative_signal_included(self):
        signals = [
            Signal(
                conversation_id="1",
                turn_index=2,
                signal_type="negative",
                confidence=0.8,
                evidence="that's wrong"
            )
        ]
        conversations = {
            "1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
                {"role": "user", "content": "That's wrong"},
            ]
        }
        pairs = to_dpo(signals, conversations)
        assert len(pairs) == 1
        assert pairs[0].chosen is None
        assert pairs[0].rejected == "Python is a snake."
        assert pairs[0].signal_type == "negative"

    def test_rephrase_signal_included(self):
        signals = [
            Signal(
                conversation_id="1",
                turn_index=2,
                signal_type="rephrase",
                confidence=0.7,
                evidence="What is Python?",
                context={"previous_query": "What is Python?"}
            )
        ]
        conversations = {
            "1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
                {"role": "user", "content": "What is Python?"},
            ]
        }
        pairs = to_dpo(signals, conversations)
        assert len(pairs) == 1
        assert pairs[0].rejected == "Python is a snake."


class TestToKto:
    def test_empty_signals(self):
        examples = to_kto([])
        assert examples == []

    def test_positive_signal(self):
        signals = [
            Signal(
                conversation_id="1",
                turn_index=2,
                signal_type="positive",
                confidence=0.8,
                evidence="thanks"
            )
        ]
        conversations = {
            "1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "Thanks! That was helpful."},
            ]
        }
        examples = to_kto(signals, conversations)
        assert len(examples) == 1
        assert examples[0].label is True
        assert examples[0].completion == "Python is a programming language."

    def test_negative_signal(self):
        signals = [
            Signal(
                conversation_id="1",
                turn_index=2,
                signal_type="negative",
                confidence=0.8,
                evidence="that's wrong"
            )
        ]
        conversations = {
            "1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
                {"role": "user", "content": "That's wrong"},
            ]
        }
        examples = to_kto(signals, conversations)
        assert len(examples) == 1
        assert examples[0].label is False
        assert examples[0].completion == "Python is a snake."


class TestToRejection:
    def test_empty_signals(self):
        pairs = to_rejection([])
        assert pairs == []

    def test_returns_dict_format(self):
        signals = [
            Signal(
                conversation_id="1",
                turn_index=2,
                signal_type="negative",
                confidence=0.8,
                evidence="that's wrong"
            )
        ]
        conversations = {
            "1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},
                {"role": "user", "content": "That's wrong"},
            ]
        }
        pairs = to_rejection(signals, conversations)
        assert len(pairs) == 1
        assert isinstance(pairs[0], dict)
        assert "prompt" in pairs[0]
        assert "rejected" in pairs[0]
        assert "signal_type" in pairs[0]
        assert "confidence" in pairs[0]


class TestEdgeCases:
    def test_unicode_conversation_dpo(self):
        """DPO converter handles Unicode conversation text."""
        signals = [
            Signal(
                conversation_id="zh-1",
                turn_index=2,
                signal_type="negative",
                confidence=0.8,
                evidence="不对"
            )
        ]
        conversations = {
            "zh-1": [
                {"role": "user", "content": "Python是什么？"},
                {"role": "assistant", "content": "Python是一种蛇。"},
                {"role": "user", "content": "不对"},
            ]
        }
        pairs = to_dpo(signals, conversations)
        assert len(pairs) == 1
        assert "Python是一种蛇" in pairs[0].rejected

    def test_unicode_conversation_kto(self):
        """KTO converter handles Unicode text."""
        signals = [
            Signal(
                conversation_id="zh-1",
                turn_index=2,
                signal_type="positive",
                confidence=0.8,
                evidence="谢谢"
            )
        ]
        conversations = {
            "zh-1": [
                {"role": "user", "content": "Python是什么？"},
                {"role": "assistant", "content": "Python是一种编程语言。"},
                {"role": "user", "content": "谢谢！很有帮助。"},
            ]
        }
        examples = to_kto(signals, conversations)
        assert len(examples) == 1
        assert examples[0].label is True

    def test_long_prompt_truncation(self):
        """Very long prompts are safely handled."""
        long_text = "Python is a programming language. " * 500
        signals = [
            Signal(
                conversation_id="long-1",
                turn_index=2,
                signal_type="negative",
                confidence=0.8,
                evidence="wrong"
            )
        ]
        conversations = {
            "long-1": [
                {"role": "user", "content": long_text},
                {"role": "assistant", "content": long_text},
                {"role": "user", "content": "That's wrong"},
            ]
        }
        pairs = to_dpo(signals, conversations)
        assert len(pairs) == 1
        # rejected is truncated to 2000 chars
        assert len(pairs[0].rejected) <= 2000

    def test_abandon_signal_to_dpo(self):
        """Abandon signals produce valid DPO pairs."""
        signals = [
            Signal(
                conversation_id="1",
                turn_index=2,
                signal_type="abandon",
                confidence=0.85,
                evidence="never mind"
            )
        ]
        conversations = {
            "1": [
                {"role": "user", "content": "Help me"},
                {"role": "assistant", "content": "Sure, what do you need?"},
                {"role": "user", "content": "Never mind"},
            ]
        }
        pairs = to_dpo(signals, conversations)
        assert len(pairs) == 1
        assert pairs[0].rejected == "Sure, what do you need?"


class TestChosenStrategy:
    """Tests for DPO chosen-response inference."""

    def test_subsequent_positive_finds_chosen(self):
        """When user corrects then later says thanks, chosen is inferred."""
        from sigx.converters.preference import CHOSEN_SUBSEQUENT

        signals = [
            Signal("c1", 2, "correction", 0.85, "Actually I meant..."),
            Signal("c1", 4, "positive", 0.8, "Thanks!"),
        ]
        conversations = {
            "c1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},           # turn 1 (rejected)
                {"role": "user", "content": "Actually I meant the language."},   # turn 2 (correction)
                {"role": "assistant", "content": "Python is a programming language."},  # turn 3 (chosen!)
                {"role": "user", "content": "Thanks!"},                          # turn 4 (positive)
            ],
        }
        pairs = to_dpo(signals, conversations, chosen_strategy=CHOSEN_SUBSEQUENT)
        assert len(pairs) == 1
        assert pairs[0].rejected == "Python is a snake."
        assert pairs[0].chosen == "Python is a programming language."
        assert pairs[0].signal_type == "correction"

    def test_no_positive_falls_back_to_last_assistant(self):
        """Without positive signal, fall back to last assistant response."""
        from sigx.converters.preference import CHOSEN_SUBSEQUENT

        signals = [
            Signal("c1", 2, "negative", 0.8, "That's wrong"),
        ]
        conversations = {
            "c1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},     # turn 1 (rejected)
                {"role": "user", "content": "That's wrong"},                # turn 2 (negative)
                {"role": "assistant", "content": "Python is a programming language."},  # turn 3 (chosen fallback)
            ],
        }
        pairs = to_dpo(signals, conversations, chosen_strategy=CHOSEN_SUBSEQUENT)
        assert len(pairs) == 1
        assert pairs[0].rejected == "Python is a snake."
        assert pairs[0].chosen == "Python is a programming language."

    def test_further_negative_blocks_fallback(self):
        """If user complains again after correction, fallback is blocked."""
        from sigx.converters.preference import CHOSEN_SUBSEQUENT

        signals = [
            Signal("c1", 2, "correction", 0.85, "No I meant..."),
            Signal("c1", 4, "negative", 0.8, "Still wrong"),
        ]
        conversations = {
            "c1": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a snake."},           # turn 1 (rejected)
                {"role": "user", "content": "No, I meant the language."},         # turn 2 (correction)
                {"role": "assistant", "content": "Python is a reptile."},         # turn 3 (also bad?)
                {"role": "user", "content": "Still wrong"},                       # turn 4 (negative again)
                {"role": "assistant", "content": "Python is a programming language."},  # turn 5
            ],
        }
        pairs = to_dpo(signals, conversations, chosen_strategy=CHOSEN_SUBSEQUENT)
        # First signal (turn 2): can't fall back past turn 4 negative
        pair_t2 = [p for p in pairs if p.signal_type == "correction"]
        assert len(pair_t2) == 1
        # chosen should NOT be turn 5 (blocked by turn 4 complaint)
        assert pair_t2[0].chosen is None

    def test_chosen_none_strategy(self):
        """'none' strategy always yields chosen=None."""
        from sigx.converters.preference import CHOSEN_NONE

        signals = [
            Signal("c1", 2, "negative", 0.8, "wrong"),
            Signal("c1", 4, "positive", 0.8, "thanks"),
        ]
        conversations = {
            "c1": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "bad"},     # turn 1
                {"role": "user", "content": "wrong"},        # turn 2
                {"role": "assistant", "content": "good"},    # turn 3
                {"role": "user", "content": "thanks"},       # turn 4
            ],
        }
        pairs = to_dpo(signals, conversations, chosen_strategy=CHOSEN_NONE)
        assert len(pairs) == 1
        assert pairs[0].chosen is None

    def test_last_assistant_strategy(self):
        """'last_assistant' always picks the final assistant response."""
        from sigx.converters.preference import CHOSEN_LAST_ASSISTANT

        signals = [
            Signal("c1", 2, "negative", 0.8, "wrong"),
        ]
        conversations = {
            "c1": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "bad"},
                {"role": "user", "content": "wrong"},
                {"role": "assistant", "content": "final answer"},
            ],
        }
        pairs = to_dpo(signals, conversations, chosen_strategy=CHOSEN_LAST_ASSISTANT)
        assert len(pairs) == 1
        assert pairs[0].chosen == "final answer"
