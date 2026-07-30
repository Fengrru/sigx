"""
SigX - Implicit Feedback Signal Extraction for LLM Alignment.

A lightweight library that extracts structured training signals from
raw conversation logs, converting implicit user feedback (rephrases,
corrections, sentiment, abandonment) into DPO/KTO training data
compatible with TRL, VeRL, and other RLHF frameworks.

Example:
    >>> from sigx import Pipeline, RephraseDetector, SentimentDetector
    >>> pipeline = Pipeline([
    ...     RephraseDetector(similarity_threshold=0.6),
    ...     SentimentDetector(min_confidence=0.6),
    ... ])
    >>> signals = pipeline.run(conversations)
    >>> pairs = pipeline.to_dpo(conversations)
"""

from .converters import (
    CHOSEN_LAST_ASSISTANT,
    CHOSEN_NONE,
    CHOSEN_SUBSEQUENT,
    to_dpo,
    to_kto,
    to_rejection,
)
from .exceptions import (
    BenchmarkError,
    ConfigurationError,
    ConversationFormatError,
    DataLoadingError,
    ExtractionError,
    LLMConnectionError,
    LLMResponseError,
    QualityGateError,
    SigXError,
)
from .extractors import (
    AbandonDetector,
    BaseExtractor,
    LLMExtractor,
    RephraseDetector,
    SentimentDetector,
)
from .filters import QualityGate
from .io import load_conversations, load_wildchat, stream_wildchat
from .pipeline import Pipeline
from .types import (
    KTOExample,
    PreferencePair,
    Signal,
    SignalReport,
    generate_report,
    save_pairs_jsonl,
    save_signals_jsonl,
)

__version__ = "0.1.0"
__all__ = [
    "Pipeline",
    "Signal",
    "PreferencePair",
    "KTOExample",
    "SignalReport",
    "generate_report",
    "save_signals_jsonl",
    "save_pairs_jsonl",
    "RephraseDetector",
    "SentimentDetector",
    "AbandonDetector",
    "LLMExtractor",
    "BaseExtractor",
    "QualityGate",
    "to_dpo",
    "to_kto",
    "to_rejection",
    "CHOSEN_NONE",
    "CHOSEN_SUBSEQUENT",
    "CHOSEN_LAST_ASSISTANT",
    "load_conversations",
    "load_wildchat",
    "stream_wildchat",
    "SigXError",
    "ExtractionError",
    "ConversationFormatError",
    "ConfigurationError",
    "LLMConnectionError",
    "LLMResponseError",
    "DataLoadingError",
    "BenchmarkError",
    "QualityGateError",
]
