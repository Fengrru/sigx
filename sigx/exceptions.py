"""
Custom exceptions for the SigX library.

This module defines exception classes for better error handling
and debugging across the SigX codebase.
"""


class SigXError(Exception):
    """Base exception class for all SigX errors."""
    pass


class ExtractionError(SigXError):
    """Raised when signal extraction fails."""
    pass


class ConversationFormatError(SigXError):
    """Raised when conversation data has invalid format."""
    pass


class ConfigurationError(SigXError):
    """Raised when configuration parameters are invalid."""
    pass


class LLMConnectionError(SigXError):
    """Raised when LLM API connection fails."""
    pass


class LLMResponseError(SigXError):
    """Raised when LLM response cannot be parsed or is invalid."""
    pass


class DataLoadingError(SigXError):
    """Raised when data loading fails."""
    pass


class BenchmarkError(SigXError):
    """Raised when benchmark evaluation fails."""
    pass


class QualityGateError(SigXError):
    """Raised when quality gate filtering fails."""
    pass
