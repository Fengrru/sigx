"""Tests for custom exceptions."""
import pytest

from sigx.exceptions import (
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


class TestExceptionHierarchy:
    def test_all_inherit_from_sigx_error(self):
        """All custom exceptions inherit from SigXError."""
        exceptions = [
            ExtractionError,
            ConversationFormatError,
            ConfigurationError,
            LLMConnectionError,
            LLMResponseError,
            DataLoadingError,
            BenchmarkError,
            QualityGateError,
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, SigXError)

    def test_sigx_error_inherits_from_exception(self):
        """SigXError inherits from Exception."""
        assert issubclass(SigXError, Exception)

    def test_exception_can_be_raised_and_caught(self):
        """Custom exceptions can be raised and caught."""
        with pytest.raises(SigXError):
            raise SigXError("test error")

        with pytest.raises(DataLoadingError):
            raise DataLoadingError("file not found")

        with pytest.raises(BenchmarkError):
            raise BenchmarkError("invalid benchmark")

    def test_exception_messages(self):
        """Custom exceptions preserve error messages."""
        msg = "Something went wrong"
        exc = SigXError(msg)
        assert str(exc) == msg

        exc = ConfigurationError(msg)
        assert str(exc) == msg
