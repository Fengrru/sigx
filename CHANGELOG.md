# Changelog

All notable changes to SigX will be documented in this file.

## [0.1.0] - 2026-07-30

### Added
- **Pipeline**: Core orchestration class for extract → filter → convert workflow.
- **Extractors**:
  - `RephraseDetector`: TF-IDF cosine similarity detection of user re-asking.
  - `SentimentDetector`: Regex-based (41 patterns) with optional ML mode (LogisticRegression).
  - `AbandonDetector`: Frustration pattern matching + trailing-assistant heuristic.
  - `LLMExtractor`: OpenAI-compatible LLM classifier (GPT-4o-mini, vLLM, Ollama, etc.).
- **Converters**:
  - `to_dpo()`: DPO preference pairs with smart chosen inference (subsequent/last_assistant/none).
  - `to_kto()`: KTO binary-labeled examples.
  - `to_rejection()`: Rejection-sampling format.
- **IO**: `load_conversations()`, `load_wildchat()`, `stream_wildchat()`.
- **Evaluation**: `pipeline.evaluate()` with per-type precision/recall/F1 metrics.
- **Benchmark**: 20-conversation labeled dataset (`tests/benchmark.json`).
- **Tests**: 62 tests across extractors, converters, filters, and pipeline.
- **Documentation**: Professional README with quick start, API reference, and research background.
