"""
RephraseDetector — detects when a user rephrases or repeats a question.

When a user asks the same or similar question again after receiving
a model response, it implies the previous response was unsatisfactory.
This transforms that implicit signal into a structured rejection label.
"""

import logging
import re
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..types import Signal
from .base import BaseExtractor

logger = logging.getLogger(__name__)

_NON_CONTENT_RE = re.compile(
    r"^(thanks|thank you|thx|ok|okay|got it|i see|great|perfect)\b",
    re.IGNORECASE,
)


class RephraseDetector(BaseExtractor):
    """
    Detect when a user rephrases a previous question.

    Uses TF-IDF cosine similarity to compare consecutive user turns.
    High similarity between two user turns (skipping the assistant turn
    between them) suggests the assistant's response was unhelpful.

    Args:
        similarity_threshold: Cosine similarity above which two turns
            are considered a rephrase. Default 0.6.
        min_turn_length: Minimum character length for a user turn to
            be considered. Shorter turns (e.g., "thanks") are filtered.
        skip_acknowledgments: If True, skip short acknowledgment turns
            like "thanks", "ok", etc. when comparing.
    """

    name = "rephrase"

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        min_turn_length: int = 20,
        skip_acknowledgments: bool = True,
        max_features: int = 5000,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_turn_length = min_turn_length
        self.skip_acknowledgments = skip_acknowledgments
        self.max_features = max_features

    def extract(self, conversation: Dict) -> List[Signal]:
        turns = conversation.get("conversation", [])
        conv_id = conversation.get("conversation_id", "")

        if not turns:
            return []

        user_turns = [
            (i, t.get("content", "").strip())
            for i, t in enumerate(turns)
            if t.get("role") in ("user", "human")
        ]
        if len(user_turns) < 2:
            return []

        pairs = self._filter_turns(user_turns)
        if len(pairs) < 2:
            return []

        texts = [t for _, t in pairs]
        # Create a fresh vectorizer per call to avoid state contamination
        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        try:
            all_texts = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        signals: List[Signal] = []

        for idx in range(1, len(pairs)):
            prev_turn_idx, prev_text = pairs[idx - 1]
            curr_turn_idx, curr_text = pairs[idx]

            try:
                sim = float(
                    cosine_similarity(all_texts[idx - 1 : idx], all_texts[idx : idx + 1])[0][0]
                )
            except Exception:
                continue

            if sim >= self.similarity_threshold:
                assistant_turn = self._get_assistant_between(
                    turns, prev_turn_idx, curr_turn_idx
                )

                signals.append(
                    Signal(
                        conversation_id=conv_id,
                        turn_index=curr_turn_idx,
                        signal_type="rephrase",
                        confidence=_scale_confidence(sim, self.similarity_threshold),
                        evidence=curr_text[:500],
                        context={
                            "previous_query": prev_text[:500],
                            "rejected_response": (
                                assistant_turn.get("content", "")[:500]
                                if assistant_turn
                                else ""
                            ),
                            "similarity": round(sim, 4),
                            "method": "tfidf",
                        },
                    )
                )

        logger.debug("RephraseDetector extracted %d signals from conv %s", len(signals), conv_id)
        return signals

    def _filter_turns(self, user_turns: List[tuple]) -> List[tuple]:
        pairs = []
        for orig_idx, text in user_turns:
            cleaned = text.strip()
            if len(cleaned) < self.min_turn_length:
                if self.skip_acknowledgments and _NON_CONTENT_RE.match(cleaned):
                    continue
            pairs.append((orig_idx, cleaned))
        return pairs

    @staticmethod
    def _get_assistant_between(
        turns: List[Dict], start: int, end: int
    ) -> Dict | None:
        for i in range(start + 1, end):
            if turns[i].get("role") in ("assistant", "gpt", "model"):
                return turns[i]
        return None


def _scale_confidence(similarity: float, threshold: float) -> float:
    """Map similarity to [0, 1] confidence, saturating above threshold."""
    if similarity >= threshold:
        return float(np.clip((similarity - threshold) / (1.0 - threshold) * 0.6 + 0.4, 0.0, 1.0))
    return float(np.clip(similarity / threshold * 0.3, 0.0, 1.0))
