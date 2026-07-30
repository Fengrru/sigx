# SigX

**Implicit Feedback Signal Extraction for LLM Alignment**

SigX converts raw conversation logs into structured training data for LLM alignment. It automatically identifies implicit user feedback — rephrases, corrections, dissatisfaction, satisfaction, and abandonment — and transforms these signals into DPO and KTO training formats.

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [API Reference](api/pipeline.md)
- [Examples](../examples/)
- [Contributing](contributing/development.md)

## Why SigX?

Traditional alignment relies on expensive human annotations or synthetic preferences from GPT-4. In reality, every chat log already contains rich preference signals — users re-ask questions when unsatisfied, say "actually I meant..." when correcting, say "thanks!" when pleased, and abandon conversations when frustrated. SigX mines these signals at zero marginal cost.
