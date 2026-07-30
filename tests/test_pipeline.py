"""Tests for pipeline module."""

from sigx.extractors import RephraseDetector, SentimentDetector
from sigx.filters import QualityGate
from sigx.pipeline import Pipeline


class TestPipeline:
    def test_empty_conversations(self):
        pipeline = Pipeline(
            extractors=[RephraseDetector()], quality_gate=QualityGate(min_confidence=0.5)
        )
        signals = pipeline.run([])
        assert signals == []

    def test_single_conversation(self):
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                    {"role": "user", "content": "Thanks! That was helpful."},
                ],
            }
        ]
        signals = pipeline.run(convos)
        assert len(signals) >= 1

    def test_multiple_extractors(self):
        pipeline = Pipeline(
            extractors=[
                RephraseDetector(similarity_threshold=0.3),
                SentimentDetector(min_confidence=0.5),
            ],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python programming language?"},
                    {"role": "assistant", "content": "Python is a snake."},
                    {
                        "role": "user",
                        "content": "What is Python programming language and how to use it?",
                    },
                ],
            }
        ]
        signals = pipeline.run(convos)
        assert len(signals) >= 1

    def test_to_dpo(self):
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a snake."},
                    {"role": "user", "content": "That's not what I asked"},
                ],
            }
        ]
        pairs = pipeline.to_dpo(convos)
        assert len(pairs) >= 1
        assert pairs[0].chosen is None

    def test_to_kto(self):
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                    {"role": "user", "content": "Thanks! That was helpful."},
                ],
            }
        ]
        examples = pipeline.to_kto(convos)
        assert len(examples) >= 1

    def test_return_report(self):
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                    {"role": "user", "content": "Thanks! That was helpful."},
                ],
            }
        ]
        signals, report = pipeline.run(convos, return_report=True)
        assert "conversations" in report
        assert "total_turns" in report
        assert "raw_signals" in report
        assert "filtered_signals" in report

    def test_report_method(self):
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                    {"role": "user", "content": "Thanks! That was helpful."},
                ],
            }
        ]
        report_text = pipeline.report(convos)
        assert "SigX Signal Extraction Report" in report_text
        assert "Conversations:" in report_text

    def test_to_kto_return_report(self):
        """to_kto with return_report=True returns a non-empty report dict."""
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                    {"role": "user", "content": "Thanks! That was helpful."},
                ],
            }
        ]
        examples, report = pipeline.to_kto(convos, return_report=True)
        assert isinstance(examples, list)
        assert isinstance(report, dict)
        # Report should contain real data, not be empty
        assert "conversations" in report
        assert report["conversations"] == 1

    def test_unicode_conversation(self):
        """Pipeline handles Chinese/Unicode conversations."""
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "zh-1",
                "conversation": [
                    {"role": "user", "content": "Python是什么？"},
                    {"role": "assistant", "content": "Python是一种编程语言。"},
                    {"role": "user", "content": "谢谢，很有帮助！"},
                ],
            }
        ]
        signals = pipeline.run(convos)
        assert isinstance(signals, list)
        # Must not crash; Chinese text may produce 0 signals but that's fine

    def test_many_conversations(self):
        """Pipeline handles a batch of conversations."""
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = []
        for i in range(20):
            convos.append(
                {
                    "conversation_id": str(i),
                    "conversation": [
                        {"role": "user", "content": "What is Python?"},
                        {"role": "assistant", "content": "Python is a programming language."},
                        {"role": "user", "content": "That's wrong"},
                    ],
                }
            )
        signals = pipeline.run(convos)
        assert len(signals) >= 1

    def test_to_dpo_return_report(self):
        """to_dpo with return_report=True returns stats."""
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )
        convos = [
            {
                "conversation_id": "1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a snake."},
                    {"role": "user", "content": "That's not what I asked"},
                ],
            }
        ]
        pairs, report = pipeline.to_dpo(convos, return_report=True)
        assert isinstance(pairs, list)
        assert "conversations" in report

    def test_evaluate_with_benchmark_list(self):
        """evaluate() works with a list of benchmark items."""
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.3),
        )
        benchmark = [
            {
                "conversation_id": "b1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a snake."},
                    {"role": "user", "content": "That's wrong"},
                ],
                "ground_truth": [{"turn_index": 2, "signal_type": "negative"}],
            },
            {
                "conversation_id": "b2",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                    {"role": "user", "content": "Thanks!"},
                ],
                "ground_truth": [{"turn_index": 2, "signal_type": "positive"}],
            },
        ]
        metrics = pipeline.evaluate(benchmark)
        assert "overall" in metrics
        assert "per_type" in metrics
        assert "f1" in metrics["overall"]
        assert 0.0 <= metrics["overall"]["f1"] <= 1.0

    def test_evaluate_with_file_path(self):
        """evaluate() accepts a JSON file path."""
        import os

        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.3),
        )
        benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark.json")
        metrics = pipeline.evaluate(benchmark_path)
        assert "overall" in metrics
        assert "per_type" in metrics
        # Benchmark has 20 items with ground truth
        assert metrics["overall"]["support"] > 0

    def test_evaluate_empty(self):
        """evaluate() handles empty benchmark gracefully."""
        pipeline = Pipeline(extractors=[SentimentDetector()])
        metrics = pipeline.evaluate([])
        assert metrics["overall"]["f1"] == 0.0
        assert metrics["overall"]["support"] == 0

    def test_evaluate_reports_per_type(self):
        """evaluate() breaks down metrics by signal type."""
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.3),
        )
        benchmark = [
            {
                "conversation_id": "b1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a snake."},
                    {"role": "user", "content": "That's wrong, incorrect answer."},
                ],
                "ground_truth": [{"turn_index": 2, "signal_type": "negative"}],
            },
        ]
        metrics = pipeline.evaluate(benchmark)
        assert "negative" in metrics["per_type"]
