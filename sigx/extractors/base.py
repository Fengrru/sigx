from abc import ABC, abstractmethod
from typing import Dict, List

from ..types import Signal


class BaseExtractor(ABC):
    """
    Base class for all signal extractors.

    This abstract class defines the interface that all signal extractors
    must implement. Each extractor takes a conversation dictionary and
    returns a list of Signal objects representing detected implicit
    feedback signals.

    Attributes:
        name: Unique identifier for this extractor type.

    Example:
        >>> class MyExtractor(BaseExtractor):
        ...     name = "my_extractor"
        ...
        ...     def extract(self, conversation: Dict) -> List[Signal]:
        ...         signals = []
        ...         # Your detection logic here
        ...         return signals
    """

    name: str = "base"

    @abstractmethod
    def extract(self, conversation: Dict) -> List[Signal]:
        """
        Extract signals from a single conversation.

        This method must be implemented by all subclasses. It should
        analyze the conversation and return a list of detected signals.

        Args:
            conversation: A dictionary containing the conversation data.
                Expected format:
                {
                    "conversation_id": str,
                    "conversation": [
                        {"role": "user"|"assistant", "content": str},
                        ...
                    ]
                }

        Returns:
            A list of Signal objects representing detected feedback signals.
        """
        ...

    def __call__(self, conversation: Dict) -> List[Signal]:
        """
        Make the extractor callable.

        This allows using the extractor as a function:
            signal = extractor(conversation)

        Args:
            conversation: A dictionary containing the conversation data.

        Returns:
            A list of Signal objects representing detected feedback signals.
        """
        return self.extract(conversation)
