"""
AbandonDetector — detect conversations where the user likely gave up.

Signals of abandonment include:
- The user stops responding after a long or confusing assistant reply.
- The last user message is a frustrated pattern ("never mind", "forget it").
- The conversation ends without resolution on the user's side.
"""

import logging
import re
from typing import Dict, List

from ..types import Signal
from .base import ASSISTANT_ROLES, USER_ROLES, BaseExtractor, format_history

logger = logging.getLogger(__name__)


FRUSTRATION_PATTERNS = [
    (re.compile(p, re.IGNORECASE), w)
    for p, w in [
        (r"\b(never\s+mind|nevermind)\b", 0.85),
        (r"\b(forget\s+it|forget\s+about\s+it)\b", 0.85),
        (r"\b(nvm|whatever)\b", 0.7),
        (r"\b(i['’]ll\s+(just\s+)?(figure|google|search|look)\s+it)\b", 0.75),
        (r"\b(this\s+is\s+(going\s+)?nowhere|not\s+going\s+anywhere)\b", 0.9),
        (r"\b(i\s+give\s+up)\b", 0.9),
        (r"\b(i['’]m\s+done|i\s+quit|this\s+is\s+useless|waste\s+of\s+time)\b", 0.85),
        (r"\b(can['’]t\s+help\s+me|you['’]re\s+not\s+helping|not\s+helping)\b", 0.8),
    ]
]


class AbandonDetector(BaseExtractor):
    """
    Detect implicit conversation abandonment signals.

    Checks if the last user turn shows frustration or if the assistant
    left a question hanging (conversation ended on assistant turn).

    Args:
        min_assistant_length: Assistant response length (chars) above which
            a trailing assistant turn is considered a possible abandonment.
        min_turns: Minimum conversation turns to check for abandonment.
        require_unanswered_question: If True, only flag conversation_ends_on_assistant
            when the last user turn was a question (ends with '?').
    """

    name = "abandon"

    def __init__(
        self,
        min_assistant_length: int = 300,
        min_turns: int = 3,
        require_unanswered_question: bool = True,
    ):
        self.min_assistant_length = min_assistant_length
        self.min_turns = min_turns
        self.require_unanswered_question = require_unanswered_question

    def extract(self, conversation: Dict) -> List[Signal]:
        turns = conversation.get("conversation", [])
        conv_id = conversation.get("conversation_id", "")

        if len(turns) < self.min_turns:
            return []

        signals: List[Signal] = []

        # Signal 1: last turn is user with frustration pattern
        last_turn = turns[-1]
        if last_turn.get("role") in USER_ROLES:
            text = last_turn.get("content", "").strip()
            for pattern, weight in FRUSTRATION_PATTERNS:
                if pattern.search(text):
                    assistant_text = ""
                    prompt_up_to = len(turns) - 1
                    if len(turns) >= 2 and turns[-2].get("role") in ASSISTANT_ROLES:
                        assistant_text = turns[-2].get("content", "")
                        prompt_up_to = len(turns) - 2
                    signals.append(
                        Signal(
                            conversation_id=conv_id,
                            turn_index=len(turns) - 1,
                            signal_type="abandon",
                            confidence=weight,
                            evidence=text[:500],
                            context={
                                "matched_pattern": pattern.pattern[:100],
                                "last_assistant_response": assistant_text[:500],
                                "rejected_response": assistant_text[:500],
                                "conversation_prompt": format_history(turns, prompt_up_to),
                                "position": "last_user_turn",
                            },
                        )
                    )
                    break

        # Signal 2: conversation ends on assistant turn, no user follow-up.
        # Only trigger if the last user turn was an unanswered question.
        if last_turn.get("role") in ASSISTANT_ROLES:
            text = last_turn.get("content", "").strip()
            if len(text) >= self.min_assistant_length:
                # Find the last user turn before this assistant response
                last_user_turn = None
                for _idx, t in enumerate(reversed(turns[:-1])):
                    if t.get("role") in USER_ROLES:
                        last_user_turn = t
                        break

                if last_user_turn is None:
                    return signals

                user_text = last_user_turn.get("content", "").strip()

                # If require_unanswered_question, only flag when user asked a question
                # that was left unanswered (stronger abandonment signal).
                if self.require_unanswered_question and not user_text.endswith("?"):
                    return signals

                # Higher confidence if the user explicitly asked a question
                confidence = 0.55 if user_text.endswith("?") else 0.35

                signals.append(
                    Signal(
                        conversation_id=conv_id,
                        turn_index=len(turns) - 1,
                        signal_type="abandon",
                        confidence=confidence,
                        evidence=text[:500],
                        context={
                            "reason": "conversation_ends_on_assistant",
                            "assistant_length": len(text),
                            "total_turns": len(turns),
                            "last_user_question": user_text.endswith("?"),
                            "rejected_response": text[:500],
                            "conversation_prompt": format_history(turns, len(turns) - 1),
                        },
                    )
                )

        logger.debug("AbandonDetector extracted %d signals from conv %s", len(signals), conv_id)
        return signals
