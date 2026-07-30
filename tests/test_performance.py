"""Performance tests for SigX pipeline."""

import time

from sigx import AbandonDetector, Pipeline, RephraseDetector, SentimentDetector
from sigx.filters import QualityGate


def _generate_conversations(n: int) -> list:
    """Generate n test conversations."""
    conversations = []
    for i in range(n):
        conversations.append(
            {
                "conversation_id": f"perf-{i}",
                "conversation": [
                    {"role": "user", "content": f"What is topic {i}?"},
                    {
                        "role": "assistant",
                        "content": f"Topic {i} is a subject about various things.",
                    },
                    {"role": "user", "content": f"Can you explain topic {i} in more detail?"},
                    {
                        "role": "assistant",
                        "content": f"Topic {i} encompasses many aspects including theory and practice.",
                    },
                    {"role": "user", "content": "Thanks, that helps!"},
                ],
            }
        )
    return conversations


def _generate_conversations_with_negative(n: int) -> list:
    """Generate n test conversations with negative signals for DPO conversion."""
    conversations = []
    for i in range(n):
        conversations.append(
            {
                "conversation_id": f"perf-dpo-{i}",
                "conversation": [
                    {"role": "user", "content": f"What is topic {i}?"},
                    {
                        "role": "assistant",
                        "content": f"Topic {i} is a subject about various things.",
                    },
                    {"role": "user", "content": "That's wrong, incorrect answer."},
                    {
                        "role": "assistant",
                        "content": f"Topic {i} encompasses many aspects including theory and practice.",
                    },
                ],
            }
        )
    return conversations


class TestPerformance:
    def test_pipeline_throughput_100(self):
        """Pipeline processes 100 conversations in reasonable time."""
        convos = _generate_conversations(100)
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )

        start = time.time()
        signals = pipeline.run(convos)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"100 conversations took {elapsed:.2f}s (>5s)"
        assert len(signals) > 0

    def test_pipeline_throughput_500(self):
        """Pipeline processes 500 conversations in reasonable time."""
        convos = _generate_conversations(500)
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )

        start = time.time()
        signals = pipeline.run(convos)
        elapsed = time.time() - start

        assert elapsed < 15.0, f"500 conversations took {elapsed:.2f}s (>15s)"
        assert len(signals) > 0

    def test_dpo_conversion_throughput(self):
        """DPO conversion processes 100 conversations in reasonable time."""
        convos = _generate_conversations_with_negative(100)
        pipeline = Pipeline(
            extractors=[SentimentDetector(min_confidence=0.5)],
            quality_gate=QualityGate(min_confidence=0.5),
        )

        start = time.time()
        pairs = pipeline.to_dpo(convos)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"DPO conversion took {elapsed:.2f}s (>5s)"
        assert len(pairs) > 0

    def test_multiple_extractors_throughput(self):
        """Pipeline with multiple extractors processes in reasonable time."""
        convos = _generate_conversations(100)
        pipeline = Pipeline(
            extractors=[
                SentimentDetector(min_confidence=0.5),
                RephraseDetector(similarity_threshold=0.5),
                AbandonDetector(min_turns=3),
            ],
            quality_gate=QualityGate(min_confidence=0.5),
        )

        start = time.time()
        pipeline.run(convos)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Multi-extractor pipeline took {elapsed:.2f}s (>10s)"
