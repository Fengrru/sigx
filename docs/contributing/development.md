# Development Setup

## Getting Started

```bash
git clone https://github.com/fengrru/sigx.git
cd sigx
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                          # All tests
pytest --cov=sigx               # With coverage
pytest tests/test_extractors.py # Specific module
```

## Linting and Type Checking

```bash
ruff check .                    # Lint
ruff check . --fix              # Auto-fix
mypy sigx/                      # Type check (mypy)
pyright                         # Type check (pyright)
```

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
sigx/
├── pipeline.py          # Orchestration: extract -> filter -> convert
├── types.py             # Core dataclasses: Signal, PreferencePair, KTOExample
├── exceptions.py        # Custom exception classes
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
