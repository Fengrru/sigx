<p align="center">
  <h1 align="center">SigX</h1>
  <p align="center"><strong>Implicit Feedback Signal Extraction for LLM Alignment</strong></p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/sigx/"><img src="https://img.shields.io/pypi/v/sigx.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/tests-62%20passed-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/benchmark-F1%200.83-success.svg" alt="Benchmark F1">
</p>

---

## Overview

**SigX** converts raw conversation logs into structured training data for LLM alignment. It automatically identifies implicit user feedback — rephrases, corrections, dissatisfaction, satisfaction, and abandonment — that users naturally express during conversations, then transforms these signals into DPO and KTO training formats ready for TRL, VeRL, and other RLHF frameworks.

> **Why implicit feedback?** Traditional alignment relies on expensive human annotations or synthetic preferences from GPT-4. In reality, every chat log already contains rich preference signals — users re-ask questions when unsatisfied, say "actually I meant…" when correcting, say "thanks!" when pleased, and abandon conversations when frustrated. SigX mines these signals at zero marginal cost.

Inspired by [WildFeedback (Microsoft Research, 2024)](https://arxiv.org/abs/2408.15549), [WildReward (Tsinghua KEG, 2026)](https://arxiv.org/abs/2606.20482), and the broader literature on learning from implicit preferences.

---

## Highlights

- **🔌 Pluggable Extractors** — Mix and match built-in detectors (regex, TF-IDF, ML, LLM) or write your own via `BaseExtractor`.
- **📊 Smart DPO Conversion** — `chosen` responses are automatically inferred from subsequent positive turns, producing complete (prompt, chosen, rejected) triples — not just (prompt, None, rejected).
- **🤖 LLM Classifier** — Optional `LLMExtractor` uses any OpenAI-compatible API (GPT-4o-mini, Qwen, Llama via vLLM/Ollama) for high-accuracy signal classification.
- **📈 Built-in Evaluation** — `pipeline.evaluate()` computes per-type precision/recall/F1 against labeled benchmarks.
- **🎯 Multi-Format Output** — DPO pairs, KTO examples, and rejection-sampling lists, all compatible with TRL and VeRL.
- **⚡ Lightweight Core** — Only `numpy` + `scikit-learn`. No GPU required. Optional extras for LLM and HuggingFace datasets.

---

## Installation

```bash
pip install sigx
```

With optional dependencies:

```bash
pip install sigx[wildchat]   # HuggingFace WildChat dataset support
pip install sigx[llm]        # LLMExtractor (OpenAI API support)
pip install sigx[dev]        # Development tools (pytest, ruff)
```

---

## Quick Start

```python
from sigx import Pipeline, RephraseDetector, SentimentDetector

pipeline = Pipeline([
    RephraseDetector(similarity_threshold=0.6),
    SentimentDetector(min_confidence=0.6),
])

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

# Convert to DPO training pairs (chosen auto-inferred from positive turns)
pairs = pipeline.to_dpo(conversations)
for p in pairs:
    print(f"Prompt:  {p.prompt[:80]}...")
    print(f"Rejected: {p.rejected[:80]}...")
    print(f"Chosen:   {p.chosen[:80] if p.chosen else 'None'}...")
    print("---")

# Convert to KTO (binary good/bad labels)
examples = pipeline.to_kto(conversations)
```

---

## Signal Types

SigX detects five categories of implicit feedback, each mapped to a specific training signal:

| Signal | Trigger Pattern | Training Implication |
|--------|----------------|---------------------|
| `rephrase` | User re-asks the same question (TF-IDF cosine similarity) | Previous response was unsatisfactory → `rejected` |
| `correction` | "Actually I meant…", "No, that's not what I…" | Model output was wrong → `rejected` |
| `negative` | "That's wrong", "Not helpful", "You misunderstood" | Explicit dissatisfaction → `rejected` |
| `positive` | "Thanks!", "Exactly what I needed", "Perfect" | Explicit satisfaction → `chosen` / `label=True` |
| `abandon` | "Never mind", "I give up", conversation terminates on assistant | User gave up → `rejected` |

---

## Extractors

### RephraseDetector

Compares consecutive user turns using TF-IDF cosine similarity. High similarity between two user messages (skipping the assistant response between them) indicates the user re-asked — implying the first answer was unhelpful.

```python
from sigx import RephraseDetector

detector = RephraseDetector(
    similarity_threshold=0.6,   # Cosine similarity threshold
    min_turn_length=20,         # Ignore very short turns
    skip_acknowledgments=True,  # Skip "thanks", "ok", etc.
)
```

### SentimentDetector

Rule-based pattern matching with 41 regex patterns covering corrections, negatives, positives, sarcasm, and false-positive guards. Also supports an optional ML mode using `LogisticRegression` on character n-grams.

```python
from sigx import SentimentDetector

# Rule-only mode (default, no extra deps)
detector = SentimentDetector(min_confidence=0.6)

# ML-enhanced mode
detector = SentimentDetector(use_ml=True, min_confidence=0.5)
detector.fit(
    texts=["that's wrong", "thanks!", "actually I meant..."],
    labels=["negative", "positive", "correction"],
)
```

### AbandonDetector

Detects when users give up — either via explicit frustration patterns ("never mind", "I'll figure it out") or when a conversation ends on a long assistant response with no user follow-up.

```python
from sigx import AbandonDetector

detector = AbandonDetector(
    min_assistant_length=300,           # Min chars for trailing-assistant detection
    min_turns=3,                        # Min conversation length
    require_unanswered_question=True,   # Only flag if last user msg ends with "?"
)
```

### LLMExtractor

Uses any OpenAI-compatible LLM for high-accuracy classification. Ideal when signal quality matters more than cost/latency.

```python
from sigx import LLMExtractor

detector = LLMExtractor(
    model="gpt-4o-mini",            # Or "qwen2.5-7b-instruct" via vLLM
    base_url=None,                  # Defaults to OpenAI; set for Ollama/vLLM
    api_key="sk-...",               # Or set OPENAI_API_KEY env var
    min_confidence=0.6,
)
```

### Custom Extractors

```python
from sigx.extractors import BaseExtractor
from sigx.types import Signal

class MyExtractor(BaseExtractor):
    name = "my_extractor"

    def extract(self, conversation: dict) -> list[Signal]:
        signals = []
        # Your logic here
        return signals

pipeline = Pipeline([MyExtractor()])
```

---

## DPO Chosen Strategies

A key innovation: SigX automatically infers the `chosen` response from subsequent conversation turns, solving the "implicit feedback only tells us what's BAD" problem.

| Strategy | Behavior |
|----------|----------|
| `subsequent` *(default)* | Find the first positive user turn after the negative signal; the assistant response before it becomes `chosen`. If no positive found, fall back to the last assistant response (unless blocked by further complaints). |
| `last_assistant` | Always use the final assistant response in the conversation as `chosen`. |
| `none` | `chosen=None` (backward compatible). |

```python
from sigx import Pipeline, CHOSEN_SUBSEQUENT, CHOSEN_NONE

# Default: smart chosen inference
pipeline = Pipeline(extractors=[...], chosen_strategy=CHOSEN_SUBSEQUENT)

# Backward compatible: no chosen inference
pipeline = Pipeline(extractors=[...], chosen_strategy=CHOSEN_NONE)
```

---

## Evaluation

Evaluate extraction quality against a labeled benchmark to measure precision, recall, and F1 per signal type.

```python
metrics = pipeline.evaluate("benchmark.json")
print(metrics["overall"])   # {"precision": 0.88, "recall": 0.79, "f1": 0.83, ...}
print(metrics["per_type"])  # {"negative": {...}, "positive": {...}, ...}
```

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

### Current Benchmark Results

Evaluated on a 20-conversation benchmark with regex-based extractors (no LLM):

| Signal Type | Precision | Recall | F1 | Support |
|------------|-----------|--------|-----|---------|
| abandon | 1.00 | 1.00 | **1.00** | 3 |
| rephrase | 1.00 | 1.00 | **1.00** | 3 |
| positive | 1.00 | 0.80 | **0.89** | 5 |
| correction | 0.60 | 1.00 | **0.75** | 3 |
| negative | 1.00 | 0.40 | **0.57** | 5 |
| **Overall** | **0.88** | **0.79** | **0.83** | 19 |

> The lower recall on `negative` (0.40) reflects the inherent limitation of regex-only approaches. Adding `LLMExtractor` significantly improves this category.

---

## Output Formats

### DPO (Direct Preference Optimization)

```python
pairs = pipeline.to_dpo(conversations)
# List[PreferencePair]
# PreferencePair(prompt="...", chosen="...", rejected="...", signal_type="...", confidence=0.85)
```

Compatible with TRL's `DPOTrainer` and VeRL.

### KTO (Kahneman-Tversky Optimization)

```python
examples = pipeline.to_kto(conversations)
# List[KTOExample]
# KTOExample(prompt="...", completion="...", label=True, confidence=0.80)
```

`label=True` for positive signals, `label=False` for negative/rephrase/correction/abandon.

### Rejection Sampling

```python
pairs = pipeline.to_rejection(conversations)
# List[dict] with keys: prompt, rejected, signal_type, confidence
```

---

## Data Loading

```python
from sigx import load_conversations, load_wildchat, stream_wildchat

# ShareGPT format
convos = load_conversations("sharegpt_data.json", format="sharegpt")

# OpenAI format
convos = load_conversations("openai_data.json", format="openai")

# Generic JSONL
convos = load_conversations("logs.jsonl", format="jsonl")

# WildChat from HuggingFace
convos = load_wildchat(n=1000)

# Stream WildChat (for very large datasets)
for convo in stream_wildchat(n=10000):
    signals = pipeline.run([convo])
```

---

## End-to-End Training Pipeline

SigX is designed to fit between your chat logs and your training framework:

```
Chat Logs                    SigX                           Training
┌──────────┐    ┌─────────────────────────────┐    ┌──────────────────┐
│ JSONL    │───▶│ extract → filter → convert   │───▶│ TRL DPOTrainer   │
│ ShareGPT │    │                             │    │ TRL KTOTrainer   │
│ WildChat │    │ signals → PreferencePair[]   │    │ VeRL             │
│ OpenAI   │    │ signals → KTOExample[]       │    │ Custom training  │
└──────────┘    └─────────────────────────────┘    └──────────────────┘
```

```python
from sigx import Pipeline, SentimentDetector, RephraseDetector, AbandonDetector
from sigx.io import load_wildchat

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

---

## Architecture

```
sigx/
├── pipeline.py          # Orchestration: extract → filter → convert
├── types.py             # Core dataclasses: Signal, PreferencePair, KTOExample
├── extractors/
│   ├── base.py          # Abstract BaseExtractor
│   ├── rephrase.py      # TF-IDF cosine similarity rephrase detection
│   ├── sentiment.py     # Regex + optional ML sentiment classifier
│   ├── abandon.py       # Frustration pattern + trailing-assistant detection
│   └── llm.py           # OpenAI-compatible LLM classifier
├── filters/
│   └── quality.py       # Confidence threshold, dedup, per-conv limits
├── converters/
│   └── preference.py    # DPO / KTO / rejection-sampling converters
└── io/
    └── loader.py        # ShareGPT, OpenAI, WildChat, JSONL loaders
```

---

## Research Background

SigX builds on growing evidence that implicit feedback from real user interactions produces better alignment data than synthetic alternatives:

- **WildFeedback** (Microsoft, 2024): Extracted 20K preference pairs from 148K real ChatGPT conversations. Models fine-tuned on this data significantly outperformed those trained on UltraFeedback across AlpacaEval 2, Arena-Hard, and MT-Bench.
- **WildReward** (Tsinghua KEG, 2026): Trained reward models directly from in-the-wild interactions without human-annotated preferences, achieving SOTA on cross-sample consistency and calibration.
- **IFllm** (2026): Collected 1,336 multi-turn questions and demonstrated that implicit feedback signals substantially improve reward model pairwise accuracy (55% → 64%).

The key insight across all these works: **real user feedback captures nuance that synthetic data misses**, and **the signals are already present in existing chat logs — they just need to be extracted**.

---

## Development

```bash
git clone https://github.com/fengrru/sigx.git
cd sigx
pip install -e ".[dev]"

# Run tests
pytest                          # 62 tests

# Lint
ruff check .
```

---

## Citation

```bibtex
@software{sigx2026,
  title   = {{SigX: Implicit Feedback Signal Extraction for LLM Alignment}},
  author  = {fengrru},
  license = {MIT},
  url     = {https://github.com/fengrru/sigx},
  year    = {2026},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
