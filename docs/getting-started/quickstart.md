# Quick Start

## Basic Usage

```python
from sigx import Pipeline, RephraseDetector, SentimentDetector

# Create a pipeline with extractors
pipeline = Pipeline([
    RephraseDetector(similarity_threshold=0.6),
    SentimentDetector(min_confidence=0.6),
])

# Define conversations
conversations = [
    {
        "conversation_id": "1",
        "conversation": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a type of snake."},
            {"role": "user", "content": "That's not what I asked. I meant the programming language."},
            {"role": "assistant", "content": "Python is a high-level programming language created by Guido van Rossum."},
            {"role": "user", "content": "Thanks! That's exactly what I needed."},
        ]
    }
]

# Extract signals
signals = pipeline.run(conversations)
for s in signals:
    print(f"[{s.signal_type}] confidence={s.confidence:.2f} | {s.evidence[:60]}...")
```

## Convert to Training Data

```python
# DPO preference pairs (chosen auto-inferred from positive turns)
pairs = pipeline.to_dpo(conversations)
for p in pairs:
    print(f"Prompt:   {p.prompt[:80]}...")
    print(f"Rejected: {p.rejected[:80]}...")
    print(f"Chosen:   {p.chosen[:80] if p.chosen else 'None'}...")

# KTO examples (binary good/bad labels)
examples = pipeline.to_kto(conversations)
```

## Load Data from Files

```python
from sigx import load_conversations

# ShareGPT format
convos = load_conversations("sharegpt_data.json", format="sharegpt")

# OpenAI format
convos = load_conversations("openai_data.json", format="openai")

# Generic JSONL
convos = load_conversations("logs.jsonl", format="jsonl")
```

## Evaluate Quality

```python
metrics = pipeline.evaluate("benchmark.json")
print(metrics["overall"])   # {"precision": 0.88, "recall": 0.79, "f1": 0.83}
print(metrics["per_type"])  # {"negative": {...}, "positive": {...}, ...}
```
