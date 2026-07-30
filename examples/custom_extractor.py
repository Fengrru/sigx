"""
Custom extractor example for SigX.

Demonstrates how to create a custom signal extractor
by subclassing BaseExtractor.
"""

from sigx import Pipeline
from sigx.extractors import BaseExtractor
from sigx.types import Signal


class LengthCheckerExtractor(BaseExtractor):
    """
    Example extractor that flags assistant responses that are
    too short or too long relative to the user's question.
    """

    name = "length_checker"

    def __init__(
        self,
        min_response_length: int = 10,
        max_response_length: int = 2000,
        min_confidence: float = 0.7,
    ):
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.min_confidence = min_confidence

    def extract(self, conversation: dict) -> list[Signal]:
        signals = []
        turns = conversation.get("conversation", [])
        conv_id = conversation.get("conversation_id", "")

        for i, turn in enumerate(turns):
            # Only check assistant turns
            if turn.get("role") != "assistant":
                continue

            # Must have a preceding user turn
            if i == 0 or turns[i - 1].get("role") != "user":
                continue

            response = turn.get("content", "")

            # Check for too-short responses
            if len(response) < self.min_response_length:
                confidence = min(
                    1.0, (self.min_response_length - len(response)) / self.min_response_length
                )
                if confidence >= self.min_confidence:
                    signals.append(
                        Signal(
                            conversation_id=conv_id,
                            turn_index=i,
                            signal_type="negative",
                            confidence=round(confidence, 4),
                            evidence=response[:500],
                            context={
                                "method": "length_checker",
                                "reason": "response_too_short",
                                "response_length": len(response),
                                "min_expected": self.min_response_length,
                            },
                        )
                    )

            # Check for too-long responses
            elif len(response) > self.max_response_length:
                confidence = min(
                    1.0, (len(response) - self.max_response_length) / self.max_response_length
                )
                if confidence >= self.min_confidence:
                    signals.append(
                        Signal(
                            conversation_id=conv_id,
                            turn_index=i,
                            signal_type="negative",
                            confidence=round(confidence, 4),
                            evidence=response[:500],
                            context={
                                "method": "length_checker",
                                "reason": "response_too_long",
                                "response_length": len(response),
                                "max_expected": self.max_response_length,
                            },
                        )
                    )

        return signals


def main():
    # Create pipeline with custom extractor
    pipeline = Pipeline(
        [
            LengthCheckerExtractor(
                min_response_length=10,
                max_response_length=100,
                min_confidence=0.5,
            ),
        ]
    )

    conversations = [
        {
            "conversation_id": "test-1",
            "conversation": [
                {"role": "user", "content": "What is the meaning of life?"},
                {"role": "assistant", "content": "42"},
                {"role": "user", "content": "What?"},
                {
                    "role": "assistant",
                    "content": (
                        "The meaning of life is a philosophical question that has been "
                        "debated for centuries. Different cultures, religions, and "
                        "philosophical traditions offer various perspectives on this "
                        "topic. Some suggest it's about finding happiness, others about "
                        "serving a higher purpose, and some argue that meaning itself "
                        "is something we create."
                    ),
                },
            ],
        },
    ]

    signals = pipeline.run(conversations)
    for s in signals:
        print(f"[{s.signal_type}] conf={s.confidence:.2f}")
        print(f"  Reason: {s.context.get('reason')}")
        print(f"  Length: {s.context.get('response_length')}")
        print()


if __name__ == "__main__":
    main()
