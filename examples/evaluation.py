"""
Evaluation example for SigX.

Demonstrates how to evaluate extraction quality against
a labeled benchmark dataset.
"""

import json
import os

from sigx import AbandonDetector, Pipeline, RephraseDetector, SentimentDetector


def create_sample_benchmark():
    """Create a sample benchmark file for demonstration."""
    benchmark = {
        "description": "Sample benchmark for evaluation demo",
        "version": "1.0",
        "conversations": [
            {
                "conversation_id": "bench-1",
                "conversation": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a snake."},
                    {"role": "user", "content": "That's wrong, I meant the language."},
                ],
                "ground_truth": [
                    {"turn_index": 2, "signal_type": "correction"}
                ]
            },
            {
                "conversation_id": "bench-2",
                "conversation": [
                    {"role": "user", "content": "How do I sort a list?"},
                    {"role": "assistant", "content": "Use sorted()."},
                    {"role": "user", "content": "Thanks!"},
                ],
                "ground_truth": [
                    {"turn_index": 2, "signal_type": "positive"}
                ]
            },
            {
                "conversation_id": "bench-3",
                "conversation": [
                    {"role": "user", "content": "Explain recursion."},
                    {"role": "assistant", "content": "Recursion is when a function calls itself."},
                    {"role": "user", "content": "Can you explain recursion more clearly?"},
                ],
                "ground_truth": [
                    {"turn_index": 2, "signal_type": "rephrase"}
                ]
            },
            {
                "conversation_id": "bench-4",
                "conversation": [
                    {"role": "user", "content": "Help me debug this."},
                    {"role": "assistant", "content": "Sure, share your code."},
                    {"role": "user", "content": "Never mind, I figured it out."},
                ],
                "ground_truth": [
                    {"turn_index": 2, "signal_type": "abandon"}
                ]
            },
            {
                "conversation_id": "bench-5",
                "conversation": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "ML is a subset of AI that learns from data."},
                    {"role": "user", "content": "That's incorrect."},
                ],
                "ground_truth": [
                    {"turn_index": 2, "signal_type": "negative"}
                ]
            },
        ]
    }
    return benchmark


def main():
    # Create sample benchmark
    benchmark = create_sample_benchmark()

    # Save benchmark to file
    benchmark_path = "sample_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Created benchmark file: {benchmark_path}")

    # Create pipeline
    pipeline = Pipeline([
        SentimentDetector(min_confidence=0.5),
        RephraseDetector(similarity_threshold=0.5),
        AbandonDetector(min_turns=2, require_unanswered_question=False),
    ])

    # Evaluate against benchmark list
    print("\n" + "=" * 60)
    print("Evaluating against benchmark list...")
    print("=" * 60)

    metrics = pipeline.evaluate(benchmark["conversations"])

    print("\nOverall Metrics:")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall:    {metrics['overall']['recall']:.4f}")
    print(f"  F1:        {metrics['overall']['f1']:.4f}")
    print(f"  Support:   {metrics['overall']['support']}")

    print("\nPer-Type Breakdown:")
    for signal_type, m in metrics["per_type"].items():
        print(f"  {signal_type:12s}: P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  (n={m['support']})")

    # Also evaluate from file path
    print("\n" + "=" * 60)
    print("Evaluating from file path...")
    print("=" * 60)

    metrics_from_file = pipeline.evaluate(benchmark_path)
    print(f"  Overall F1: {metrics_from_file['overall']['f1']:.4f}")

    # Cleanup
    os.remove(benchmark_path)
    print(f"\nCleaned up {benchmark_path}")


if __name__ == "__main__":
    main()
