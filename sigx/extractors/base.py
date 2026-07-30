from abc import ABC, abstractmethod
from typing import Dict, List

from ..types import Signal


class BaseExtractor(ABC):
    """
    Base class for all signal extractors.

    Each extractor takes a conversation dictionary and returns
    a list of Signal objects.
    """

    name: str = "base"

    @abstractmethod
    def extract(self, conversation: Dict) -> List[Signal]:
        """Extract signals from a single conversation."""
        ...

    def __call__(self, conversation: Dict) -> List[Signal]:
        return self.extract(conversation)
