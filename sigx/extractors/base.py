from abc import ABC, abstractmethod
from typing import Dict, List

from ..exceptions import ConversationFormatError
from ..types import Signal

USER_ROLES = ("user", "human")
ASSISTANT_ROLES = ("assistant", "gpt", "model")


def format_history(turns: List[Dict], up_to: int, max_chars: int = 4000) -> str:
    """
    Build a "User:/Assistant:" history string from turns[0:up_to].

    Used by extractors to store a training-ready prompt in
    Signal.context["conversation_prompt"], so converters can work
    even when the original conversations are no longer available.

    Args:
        turns: Conversation turn dicts.
        up_to: Exclusive end index; the turn at this index is NOT included.
        max_chars: Keep at most this many trailing characters (most
            recent context is preserved when truncating).

    Returns:
        The formatted history string (may be empty).
    """
    parts: List[str] = []
    for i in range(max(0, up_to)):
        role = turns[i].get("role", "")
        content = str(turns[i].get("content", "") or "").strip()
        if not content:
            continue
        if role in USER_ROLES:
            parts.append(f"User: {content}")
        elif role in ASSISTANT_ROLES:
            parts.append(f"Assistant: {content}")
    history = "\n".join(parts)
    return history[-max_chars:] if len(history) > max_chars else history


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

        Raises:
            ConversationFormatError: If the conversation is not a dict or
                its "conversation" field is not a list of turn dicts.
        """
        self.validate(conversation)
        return self.extract(conversation)

    @staticmethod
    def validate(conversation: Dict) -> None:
        """Validate the basic conversation structure.

        Raises:
            ConversationFormatError: If the structure is invalid.
        """
        if not isinstance(conversation, dict):
            raise ConversationFormatError(
                f"conversation must be a dict, got {type(conversation).__name__}"
            )
        turns = conversation.get("conversation", [])
        if not isinstance(turns, list):
            raise ConversationFormatError(
                f'"conversation" field must be a list, got {type(turns).__name__}'
            )
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ConversationFormatError(f"turn {i} must be a dict, got {type(turn).__name__}")
