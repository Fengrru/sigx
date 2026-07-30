"""
Data loading example for SigX.

Demonstrates how to load conversation data from various formats
and process them with the SigX pipeline.
"""

import json
import os
import tempfile

from sigx import Pipeline, SentimentDetector, load_conversations


def create_sample_data():
    """Create sample conversation files in different formats."""

    # ShareGPT format
    sharegpt_data = [
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
                {"from": "human", "value": "Thanks!"},
            ]
        }
    ]

    # OpenAI format
    openai_data = [
        {
            "messages": [
                {"role": "user", "content": "How do I read a file?"},
                {"role": "assistant", "content": "Use open() and read()."},
                {"role": "user", "content": "Perfect!"},
            ]
        }
    ]

    # Generic JSONL format
    jsonl_lines = [
        json.dumps({
            "conversation_id": "jsonl-1",
            "conversation": [
                {"role": "user", "content": "What is Docker?"},
                {"role": "assistant", "content": "Docker is a containerization platform."},
                {"role": "user", "content": "Not helpful."},
            ]
        })
    ]

    return sharegpt_data, openai_data, jsonl_lines


def main():
    sharegpt_data, openai_data, jsonl_lines = create_sample_data()

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sharegpt_data, f)
        sharegpt_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(openai_data, f)
        openai_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(jsonl_lines))
        jsonl_path = f.name

    try:
        # Create pipeline
        pipeline = Pipeline([SentimentDetector(min_confidence=0.5)])

        # Load ShareGPT format
        print("=" * 60)
        print("Loading ShareGPT format...")
        print("=" * 60)
        convos = load_conversations(sharegpt_path, format="sharegpt")
        signals = pipeline.run(convos)
        print(f"  Loaded {len(convos)} conversations")
        print(f"  Extracted {len(signals)} signals")

        # Load OpenAI format
        print("\n" + "=" * 60)
        print("Loading OpenAI format...")
        print("=" * 60)
        convos = load_conversations(openai_path, format="openai")
        signals = pipeline.run(convos)
        print(f"  Loaded {len(convos)} conversations")
        print(f"  Extracted {len(signals)} signals")

        # Load JSONL format
        print("\n" + "=" * 60)
        print("Loading JSONL format...")
        print("=" * 60)
        convos = load_conversations(jsonl_path, format="jsonl")
        signals = pipeline.run(convos)
        print(f"  Loaded {len(convos)} conversations")
        print(f"  Extracted {len(signals)} signals")

    finally:
        # Cleanup
        os.remove(sharegpt_path)
        os.remove(openai_path)
        os.remove(jsonl_path)


if __name__ == "__main__":
    main()
