"""
Basic usage example for SigX.

Demonstrates how to extract implicit feedback signals from
conversations and convert them to DPO training pairs.
"""

from sigx import AbandonDetector, Pipeline, RephraseDetector, SentimentDetector


def main():
    # Create a pipeline with multiple extractors
    pipeline = Pipeline([
        RephraseDetector(similarity_threshold=0.6),
        SentimentDetector(min_confidence=0.6),
        AbandonDetector(min_turns=3),
    ])

    # Sample conversations
    conversations = [
        {
            "conversation_id": "example-1",
            "conversation": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a type of snake."},
                {
                    "role": "user",
                    "content": "That's not what I asked. I meant the programming language.",
                },
                {
                    "role": "assistant",
                    "content": "Python is a high-level programming language created by "
                    "Guido van Rossum.",
                },
                {"role": "user", "content": "Thanks! That's exactly what I needed."},
            ],
        },
        {
            "conversation_id": "example-2",
            "conversation": [
                {"role": "user", "content": "How do I sort a list in Python?"},
                {"role": "assistant", "content": "You can use the sorted() function."},
                {"role": "user", "content": "Perfect, thanks!"},
            ]
        },
        {
            "conversation_id": "example-3",
            "conversation": [
                {"role": "user", "content": "Explain neural networks."},
                {"role": "assistant", "content": "Neural networks are computational models."},
                {"role": "user", "content": "Can you explain neural networks in more detail?"},
            ]
        },
    ]

    # Run the pipeline
    print("=" * 60)
    print("Extracting signals...")
    print("=" * 60)
    signals = pipeline.run(conversations)

    for s in signals:
        print(f"  [{s.signal_type:10s}] conv={s.conversation_id} "
              f"turn={s.turn_index} conf={s.confidence:.2f}")
        print(f"    Evidence: {s.evidence[:80]}...")
        print()

    # Convert to DPO pairs
    print("=" * 60)
    print("Converting to DPO pairs...")
    print("=" * 60)
    pairs = pipeline.to_dpo(conversations)

    for p in pairs:
        print(f"  Prompt:   {p.prompt[:80]}...")
        print(f"  Rejected: {p.rejected[:80]}...")
        print(f"  Chosen:   {p.chosen[:80] if p.chosen else 'None'}...")
        print(f"  Signal:   {p.signal_type} (conf={p.confidence:.2f})")
        print()

    # Convert to KTO examples
    print("=" * 60)
    print("Converting to KTO examples...")
    print("=" * 60)
    kto_examples = pipeline.to_kto(conversations)

    for ex in kto_examples:
        label = "positive" if ex.label else "negative"
        print(f"  [{label:8s}] conf={ex.confidence:.2f} | {ex.completion[:80]}...")

    # Generate report
    print()
    print("=" * 60)
    print("Pipeline Report:")
    print("=" * 60)
    report_text = pipeline.report(conversations)
    print(report_text)


if __name__ == "__main__":
    main()
