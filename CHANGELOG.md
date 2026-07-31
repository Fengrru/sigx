# Changelog

All notable changes to SigX will be documented in this file.

## [0.1.1] - 2026-07-31

### Fixed
- **DPO/KTO prompt leak**: the rejected assistant response is no longer included in its own prompt; prompts are now built up to (but excluding) the rejected turn.
- **Missing `conversation_id`**: conversations without an id are assigned a positional id consistently across extraction and conversion (previously produced zero pairs silently).
- **Sentiment guards**: `FALSE_POSITIVE_GUARDS` split into `POSITIVE_GUARDS` (suppress positive only) and `NEUTRAL_GUARDS` (suppress all), so "no thanks, that's not what I needed" now correctly yields a negative signal.
- **Pattern priority**: `SentimentDetector` scans all categories and keeps the highest-weight match instead of the first hit.
- **`evaluate()`**: predictions of types absent from ground truth now count as false positives; same-turn conflicts keep the highest-confidence prediction.
- **`min_turn_length`**: short user turns are now filtered as documented (logic was inverted).
- **CJK rephrase detection**: `RephraseDetector` falls back to char n-grams for Chinese/Japanese/Korean text.
- **`QualityGate.n_dropped`**: now reports the real drop count instead of always 0.
- **Loader consistency**: `load_wildchat()` raises `DataLoadingError` (matching `stream_wildchat()`).

### Changed
- **Streaming pipeline**: conversations are consumed in a single pass; only turns from signal-producing conversations are kept in memory (generators no longer materialized).
- **Unified confidence default**: `Pipeline` default gate is `QualityGate()` (0.6); rephrase similarity at threshold maps to confidence 0.6 so detected signals survive the default gate.
- **Exceptions wired**: `ConfigurationError`, `ConversationFormatError`, `ExtractionError`, `QualityGateError`, `LLMConnectionError`, `LLMResponseError` are now actually raised at the appropriate boundaries.
- **`LLMExtractor`**: `batch_size` implemented (concurrent classification), plus new `timeout`, `max_retries`, and `on_error` options.
- **Typing**: `run`/`to_dpo`/`to_kto` use `@overload` for precise return types; mypy passes clean with sklearn/datasets overrides.

### Added
- `SIGNAL_TYPES` constant exported from `sigx`.
- 22 regression tests (92 total) covering prompt leak, id normalization, guards, evaluation FP accounting, streaming, and CJK support.

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
