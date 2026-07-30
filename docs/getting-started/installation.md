# Installation

## Requirements

- Python 3.8 or later
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0

## Install from PyPI

```bash
pip install sigx
```

## Optional Dependencies

```bash
pip install sigx[wildchat]   # HuggingFace WildChat dataset support
pip install sigx[llm]        # LLMExtractor (OpenAI API support)
pip install sigx[dev]        # Development tools (pytest, ruff, mypy, pyright)
```

## Install from Source

```bash
git clone https://github.com/fengrru/sigx.git
cd sigx
pip install -e ".[dev]"
```

## Verify Installation

```python
import sigx
print(sigx.__version__)  # 0.1.0
```
