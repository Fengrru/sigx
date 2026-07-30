# Contributing to SigX

Thank you for your interest in contributing! SigX is a lightweight library for extracting implicit feedback signals from conversation logs — and we welcome improvements of all sizes.

## Getting Started

```bash
git clone https://github.com/fengrru/sigx.git
cd sigx
pip install -e ".[dev]"
```

## Development Workflow

1. **Fork** the repository and create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code** following existing conventions:
   - Keep line length ≤ 100 characters.
   - Use type annotations on all public methods.
   - Add docstrings for public APIs.
   - Match the logging style (module-level `logger = logging.getLogger(__name__)`).

3. **Run tests** before committing:

   ```bash
   pytest                          # All 62 tests
   ruff check sigx/ tests/         # Lint
   ```

4. **Commit** with a clear message following [Conventional Commits](https://www.conventionalcommits.org/):

   ```
   feat(extractors): add toxicity detector
   fix(converters): handle empty assistant turns in DPO
   docs: update installation instructions
   ```

5. **Push** and open a Pull Request against `main`.

## Adding a New Extractor

1. Create a new file under `sigx/extractors/`.
2. Subclass `BaseExtractor` and implement `extract()`:

   ```python
   from sigx.extractors.base import BaseExtractor
   from sigx.types import Signal

   class MyExtractor(BaseExtractor):
       name = "my_extractor"

       def __init__(self, threshold: float = 0.5):
           self.threshold = threshold

       def extract(self, conversation: dict) -> list[Signal]:
           signals = []
           # Your detection logic here
           return signals
   ```

3. Export from `sigx/extractors/__init__.py`.
4. Add tests under `tests/`.

## Code Style

- This project uses **ruff** for linting and import sorting. Config is in `pyproject.toml`.
- Target Python version is **3.8+**. Do not use PEP 604 (`X | None`) or PEP 585 (`list[T]`) syntax.
- Follow [PEP 8](https://peps.python.org/pep-0008/) with the exceptions configured in `[tool.ruff]`.

## Testing

- All new features must include tests.
- Test extractors against a variety of conversation structures.
- For extractor tests, use `benchmark.json` conventions: provide `conversation_id`, `conversation`, and `ground_truth`.

## Questions?

Open an [issue](https://github.com/fengrru/sigx/issues) or start a discussion!
