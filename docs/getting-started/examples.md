# Examples

## Signal Extraction

### Detecting User Dissatisfaction

```python
from sigx import Pipeline, SentimentDetector

pipeline = Pipeline([SentimentDetector(min_confidence=0.6)])

conversations = [
    {
        "conversation_id": "1",
        "conversation": [
            {"role": "user", "content": "How do I sort a list?"},
            {"role": "assistant", "content": "Use the sorted() function."},
            {"role": "user", "content": "That's wrong, it doesn't work for my case."},
        ]
    }
]

signals = pipeline.run(conversations)
# Signals: [negative (confidence=0.85)]
```

### Detecting Rephrases

```python
from sigx import Pipeline, RephraseDetector

pipeline = Pipeline([RephraseDetector(similarity_threshold=0.6)])

conversations = [
    {
        "conversation_id": "1",
        "conversation": [
            {"role": "user", "content": "Explain neural networks to me."},
            {"role": "assistant", "content": "Neural networks are computational models."},
            {"role": "user", "content": "Can you explain neural networks with examples?"},
        ]
    }
]

signals = pipeline.run(conversations)
# Signals: [rephrase (confidence=0.89)]
```

## Training Data Conversion

### DPO Pairs with Smart Chosen Inference

```python
from sigx import Pipeline, SentimentDetector, RephraseDetector, CHOSEN_SUBSEQUENT

pipeline = Pipeline(
    extractors=[SentimentDetector(), RephraseDetector()],
    chosen_strategy=CHOSEN_SUBSEQUENT,
)

conversations = [
    {
        "conversation_id": "1",
        "conversation": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a snake."},
            {"role": "user", "content": "No, the programming language."},
            {"role": "assistant", "content": "Python is a high-level programming language."},
            {"role": "user", "content": "Perfect, thanks!"},
        ]
    }
]

pairs = pipeline.to_dpo(conversations)
# pairs[0].rejected = "Python is a snake."
# pairs[0].chosen  = "Python is a high-level programming language."
```

### KTO Examples

```python
from sigx import Pipeline, SentimentDetector

pipeline = Pipeline([SentimentDetector()])

conversations = [
    {
        "conversation_id": "1",
        "conversation": [
            {"role": "user", "content": "Thanks, that helped!"},
        ]
    }
]

examples = pipeline.to_kto(conversations)
# KTOExample(prompt="...", completion="...", label=True, confidence=0.85)
```

## Data Loading

### WildChat from HuggingFace

```python
from sigx import load_wildchat, stream_wildchat

# Load first 1000 conversations
convos = load_wildchat(n=1000)

# Stream large datasets (memory efficient)
for convo in stream_wildchat(n=10000):
    signals = pipeline.run([convo])
```

## End-to-End Training Pipeline

```python
from sigx import Pipeline, SentimentDetector, RephraseDetector, AbandonDetector
from sigx.io import load_wildchat

# Configure pipeline
pipeline = Pipeline([
    SentimentDetector(min_confidence=0.6),
    RephraseDetector(similarity_threshold=0.6),
    AbandonDetector(min_turns=3),
])

# Load and process
convos = load_wildchat(n=10000)
dpo_pairs = pipeline.to_dpo(convos)

# Feed directly to TRL
from trl import DPOTrainer
trainer = DPOTrainer(model=model, train_dataset=dpo_pairs)
trainer.train()
```
