# Custom Extractors

SigX is designed to be extensible. You can create custom extractors by subclassing `BaseExtractor`.

## Creating a Custom Extractor

```python
from sigx.extractors import BaseExtractor
from sigx.types import Signal

class ToxicityExtractor(BaseExtractor):
    """Detect toxic or harmful user messages."""

    name = "toxicity"

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def extract(self, conversation: dict) -> list[Signal]:
        signals = []
        turns = conversation.get("conversation", [])
        conv_id = conversation.get("conversation_id", "")

        for i, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue

            text = turn.get("content", "")
            # Your toxicity detection logic here
            toxicity_score = self._check_toxicity(text)

            if toxicity_score >= self.min_confidence:
                signals.append(Signal(
                    conversation_id=conv_id,
                    turn_index=i,
                    signal_type="toxic",
                    confidence=toxicity_score,
                    evidence=text[:500],
                    context={"method": "custom_toxicity"},
                ))

        return signals

    def _check_toxicity(self, text: str) -> float:
        # Implement your toxicity detection logic
        # Return a score between 0.0 and 1.0
        return 0.0
```

## Registering the Extractor

```python
from sigx import Pipeline
from my_extractors import ToxicityExtractor

pipeline = Pipeline([
    ToxicityExtractor(min_confidence=0.7),
    # ... other extractors
])
```

## Best Practices

1. **Inherit from `BaseExtractor`** — This ensures compatibility with the Pipeline.
2. **Set a unique `name`** — Used for signal type identification.
3. **Validate input** — Handle malformed conversations gracefully.
4. **Include context** — Store useful metadata in `signal.context`.
5. **Write tests** — Cover edge cases and integration with Pipeline.
