# Code Style

## Formatting

- **Line length**: 100 characters max.
- **Target Python**: 3.8+ (no PEP 604 `X | None` or PEP 585 `list[T]` syntax).
- **Linting**: Ruff with rules: E, F, W, I, N, UP, B.

## Type Annotations

- All public methods must have type annotations.
- Use `typing.Optional[X]` instead of `X | None` for Python 3.8 compatibility.
- Use `typing.List[X]`, `typing.Dict[K, V]` for container types.

## Docstrings

All public classes and methods must have docstrings. Use Google-style docstrings:

```python
def my_function(param1: str, param2: int = 0) -> bool:
    """
    Short description of the function.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.

    Example:
        >>> result = my_function("hello", 42)
        >>> print(result)
    """
```

## Logging

Use module-level loggers:

```python
import logging
logger = logging.getLogger(__name__)
```

## Conventional Commits

```
feat(extractors): add toxicity detector
fix(converters): handle empty assistant turns in DPO
docs: update installation instructions
test: add benchmark tests for LLMExtractor
refactor: simplify quality gate dedup logic
```
