# Training Pipeline Guide

SigX is designed to fit between your chat logs and your training framework.

## Architecture

```
Chat Logs                    SigX                           Training
+----------+    +-----------------------------+    +------------------+
| JSONL    |--->| extract -> filter -> convert |--->| TRL DPOTrainer   |
| ShareGPT |    |                             |    | TRL KTOTrainer   |
| WildChat |    | signals -> PreferencePair[] |    | VeRL             |
| OpenAI   |    | signals -> KTOExample[]     |    | Custom training  |
+----------+    +-----------------------------+    +------------------+
```

## End-to-End with TRL

```python
from sigx import Pipeline, SentimentDetector, RephraseDetector, AbandonDetector
from sigx.io import load_wildchat

# 1. Configure pipeline
pipeline = Pipeline([
    SentimentDetector(min_confidence=0.6),
    RephraseDetector(similarity_threshold=0.6),
    AbandonDetector(min_turns=3),
])

# 2. Load conversation data
convos = load_wildchat(n=10000)

# 3. Convert to DPO training pairs
dpo_pairs = pipeline.to_dpo(convos)

# 4. Feed to TRL
from trl import DPOTrainer
trainer = DPOTrainer(model=model, train_dataset=dpo_pairs)
trainer.train()
```

## KTO Training

```python
# Convert to KTO format
kto_examples = pipeline.to_kto(convos)

# KTO examples have binary labels
# label=True  -> desirable response
# label=False -> undesirable response
```

## Tips

1. **Tune confidence thresholds** — Higher thresholds = fewer but higher-quality signals.
2. **Use multiple extractors** — Each detector captures different feedback patterns.
3. **Evaluate against benchmarks** — Use `pipeline.evaluate()` to measure quality.
4. **Stream large datasets** — Use `stream_wildchat()` for memory-efficient processing.
