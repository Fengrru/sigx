# Evaluation Guide

SigX includes built-in evaluation to measure extraction quality against labeled benchmarks.

## Benchmark Format

Benchmark files are JSON with `conversation_id`, `conversation`, and `ground_truth` fields:

```json
{
  "conversations": [
    {
      "conversation_id": "1",
      "conversation": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a snake."},
        {"role": "user", "content": "That's wrong"}
      ],
      "ground_truth": [{"turn_index": 2, "signal_type": "negative"}]
    }
  ]
}
```

## Running Evaluation

```python
from sigx import Pipeline, SentimentDetector, RephraseDetector

pipeline = Pipeline([
    SentimentDetector(min_confidence=0.5),
    RephraseDetector(similarity_threshold=0.6),
])

# Evaluate against a benchmark file
metrics = pipeline.evaluate("benchmark.json")

print(f"Overall F1: {metrics['overall']['f1']:.4f}")
print(f"Per-type breakdown:")
for signal_type, m in metrics["per_type"].items():
    print(f"  {signal_type}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
```

## Interpreting Results

| Metric | Description |
|--------|-------------|
| **Precision** | Of all predicted signals, how many are correct? |
| **Recall** | Of all ground truth signals, how many were found? |
| **F1** | Harmonic mean of precision and recall. |
| **Support** | Number of ground truth instances for this signal type. |

## Improving Quality

1. **Add LLMExtractor** — Significantly improves recall on `negative` and `correction` types.
2. **Tune thresholds** — Lower `min_confidence` increases recall but may reduce precision.
3. **Expand patterns** — Add more regex patterns to `SentimentDetector`.
4. **Train ML mode** — Use `SentimentDetector(use_ml=True)` with labeled data.
