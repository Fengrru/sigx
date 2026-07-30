"""
SentimentDetector — hybrid detection of positive/negative implicit
feedback from user messages.

Combines rule-based regex pattern matching (fast, high precision)
with an optional ML classifier (LogisticRegression on char n-grams)
for broader coverage and calibrated confidence scores.

Pattern categories:
- correction: user explicitly corrects the assistant
- negative: user expresses dissatisfaction
- positive: user expresses satisfaction
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import Signal
from .base import BaseExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern libraries (significantly expanded)
# ---------------------------------------------------------------------------

NEGATIVE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(p, re.IGNORECASE), w)
    for p, w in [
        # Direct corrections / rejections
        (r"\bno[,.\s]+\w*\s*wrong\b", 0.90),
        (
            r"\b(not|isn['’]t|that['’]s\s+not|this\s+is\s+not)"
            r"\s+(what\s+i\s+(asked|meant|wanted|needed|said))\b",
            0.95,
        ),
        (r"\b(that['’]s\s+wrong|that\s+is\s+wrong|incorrect|you\s+are\s+wrong)\b", 0.85),
        (r"\b(try\s+again|not\s+right|you\s+misunderstood|misunderstood)\b", 0.85),
        (r"\b(not\s+helpful|doesn['’]t\s+help|not\s+useful|unhelpful)\b", 0.80),
        (r"\b(that['’]s\s+not\s+(it|correct|accurate|true|helpful|right|what\s+i))\b", 0.85),
        (
            r"\b(?<!thank\s)(you\s+(didn['’]t|do\s+not|don['’]t)"
            r"\s+(answer|understand|get|know))\b",
            0.90,
        ),
        (r"\b(that\s+doesn['’]t\s+(make\s+sense|work|answer|help))\b", 0.80),
        # Implicit dissatisfaction
        (r"\b(this\s+is\s+(wrong|incorrect|bad|terrible|awful|useless|pointless))\b", 0.85),
        (r"\b(you['’]re\s+(wrong|incorrect|mistaken|not\s+right|not\s+getting\s+it))\b", 0.85),
        (r"\b(i['’]m\s+(not\s+satisfied|disappointed|frustrated|confused))\b", 0.85),
        (r"\b(please\s+(stop|don['’]t)|stop\s+(saying|doing|giving))\b", 0.80),
        (r"\b(bad\s+(answer|response|advice|suggestion))\b", 0.85),
        (r"\b(completely\s+wrong|totally\s+wrong|absolutely\s+wrong)\b", 0.90),
        (r"\b(where\s+did\s+you\s+get\s+that)\b", 0.70),
        (r"\b(that\s+wasn['’]t\s+(my\s+question|what\s+i))\b", 0.90),
        # Sarcasm / passive-aggressive
        (r"\b(wow[,.\s]*(thanks|great|amazing|brilliant))\b", 0.65),
        (r"\b(are\s+you\s+(serious|kidding|joking))\b", 0.65),
    ]
]

CORRECTION_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(p, re.IGNORECASE), w)
    for p, w in [
        (r"\b(i\s+meant\b|what\s+i\s+meant\s+(was|is))\b", 0.85),
        (r"\b(actually[,.\s]*i)\b", 0.70),
        (r"\b(no[,.\s]+i\s+meant)\b", 0.90),
        (
            r"\b(can\s+you\s+(please\s+)?(try|redo|rephrase|"
            r"revise|rethink|restate|do\s+it\s+again))\b",
            0.75,
        ),
        (r"\b(let\s+me\s+(rephrase|clarify|be\s+more\s+specific|explain|put\s+it))\b", 0.80),
        (r"\b(that['’]s\s+not\s+(quite\s+|exactly\s+)?what\s+i)\b", 0.85),
        (r"\b(i\s+actually\s+(wanted|meant|needed|was\s+looking\s+for|was\s+asking))\b", 0.85),
        (r"\b(what\s+i\s+(really\s+)?(meant|wanted|needed|was\s+trying))\b", 0.85),
        (r"\b(i\s+should\s+have\s+(said|asked|mentioned|specified))\b", 0.80),
        (r"\b(sorry[,.\s]+(i\s+meant|let\s+me|what\s+i))\b", 0.80),
        (r"\b(to\s+be\s+(clear|precise|specific|more\s+specific))\b", 0.70),
        (r"\b(i\s+think\s+you\s+(misunderstood|misinterpreted|got\s+it\s+wrong))\b", 0.90),
        (r"\b(not\s+exactly[,.\s]+(i|what))\b", 0.80),
        (r"\b(correct\s+me\s+if\s+i['’]m\s+wrong)\b", 0.60),
    ]
]

POSITIVE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(p, re.IGNORECASE), w)
    for p, w in [
        (r"\b(thanks|thank\s+you|thx|ty)\b", 0.60),
        (r"\b(great|perfect|excellent|awesome|wonderful|fantastic|amazing|brilliant)\b", 0.70),
        (r"\b(exactly|precisely)\s*(what\s+i|!)\b", 0.80),
        (r"\b(that\s+(helps|works|is\s+helpful|is\s+great|is\s+perfect|makes\s+sense))\b", 0.85),
        (r"\b(this\s+is\s+(exactly|precisely|just)\s+what)\b", 0.90),
        (
            r"\b(very\s+helpful|super\s+helpful|really\s+helpful|"
            r"so\s+helpful|incredibly\s+helpful)\b",
            0.85,
        ),
        (r"\b(i\s+appreciate|much\s+appreciated|really\s+appreciate)\b", 0.75),
        (r"\b(well\s+done|good\s+job|nicely\s+done|great\s+job)\b", 0.80),
        (r"\b(you\s+(nailed\s+it|got\s+it|understand|rock))\b", 0.80),
        (r"\b(saved\s+me\s+(so\s+much\s+)?time)\b", 0.85),
        (r"\b(i\s+(love|like)\s+(this|it|that|your))\b", 0.75),
        (r"\b(best\s+(answer|response|explanation))\b", 0.85),
        (r"\b(finally[,.\s]+(someone|somebody|an\s+answer))\b", 0.70),
        (r"\b(this\s+(worked|solved|fixed)\s+it)\b", 0.90),
        (r"\b(you['’]re\s+(a\s+)?(lifesaver|the\s+best|amazing|awesome|a\s+legend))\b", 0.80),
    ]
]

FALSE_POSITIVE_GUARDS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(no\s+thanks|no\s+thank\s+you)\b",
        r"\b(not\s+great|not\s+perfect|not\s+exactly|not\s+really)\b",
        r"\b(not\s+helpful|not\s+useful|not\s+good)\b",
        r"\b(right\?|is\s+that\s+(right|correct)\?|am\s+i\s+(right|correct)\?)",
        r"\b(was\s+that\s+(right|correct|ok|okay)\?)",
        r"\b(i\s+don['’]t\s+think\s+so)\b",
        r"\b(i['’]m\s+not\s+sure\s+(about|if)\s+that)\b",
        r"\b(can\s+you\s+(confirm|verify|double.check))\b",
    ]
]

# ---------------------------------------------------------------------------
# ML classifier (optional, lazy-loaded sklearn)
# ---------------------------------------------------------------------------

_ML_AVAILABLE = True
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SKLPipeline
except ImportError:  # pragma: no cover
    _ML_AVAILABLE = False


class SentimentDetector(BaseExtractor):
    """
    Detect implicit positive or negative feedback from user messages.

    Uses a two-stage approach:
    1. Rule-based regex patterns (fast, high precision, always active).
    2. Optional ML classifier (broader coverage, calibrated confidence).

    Produces signals of type "correction", "negative", and "positive".

    Args:
        min_confidence: Minimum confidence to emit a signal.
        require_adjacent_assistant: If True, only emit a signal if the
            user message immediately follows an assistant message.
        use_ml: If True, enable ML classifier as fallback for turns that
            don't match any regex pattern. Requires scikit-learn.
        ml_confidence_threshold: Minimum ML confidence to emit a signal.
    """

    name = "sentiment"

    def __init__(
        self,
        min_confidence: float = 0.6,
        require_adjacent_assistant: bool = True,
        use_ml: bool = False,
        ml_confidence_threshold: float = 0.55,
    ):
        self.min_confidence = min_confidence
        self.require_adjacent_assistant = require_adjacent_assistant
        self.use_ml = use_ml
        self.ml_confidence_threshold = ml_confidence_threshold

        # ML pipeline (built lazily via fit() or first extract())
        self._ml_pipeline: Optional[SKLPipeline] = None
        self._ml_label_map: Dict[int, str] = {0: "negative", 1: "positive", 2: "correction"}

        if use_ml and not _ML_AVAILABLE:
            logger.warning(
                "use_ml=True but scikit-learn is not installed. "
                "Falling back to rule-only mode. Install with: pip install scikit-learn"
            )
            self.use_ml = False

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        **kwargs,
    ) -> "SentimentDetector":
        """
        Train the ML classifier on labeled user messages.

        Labels should be one of: "negative", "positive", "correction", "neutral".
        "neutral" examples are used as negative training signal for all classes.

        Args:
            texts: List of user message strings.
            labels: Corresponding labels for each text.
            **kwargs: Passed through to LogisticRegression (e.g. C=1.0, max_iter=500).

        Returns:
            self (for method chaining).

        Example:
            >>> detector = SentimentDetector(use_ml=True)
            >>> detector.fit(
            ...     texts=["that's wrong", "thanks!", "actually I meant..."],
            ...     labels=["negative", "positive", "correction"],
            ... )
        """
        if not _ML_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ML mode. Install with: pip install scikit-learn"
            )

        # Map string labels to integers (skip neutral)
        label_to_int = {"negative": 0, "positive": 1, "correction": 2}
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(texts, labels):
            if label in label_to_int:
                filtered_texts.append(text)
                filtered_labels.append(label_to_int[label])

        if len(set(filtered_labels)) < 2:
            logger.warning(
                "ML training requires at least 2 classes; got %d. "
                "ML mode disabled.",
                len(set(filtered_labels)),
            )
            self.use_ml = False
            return self

        # Build sklearn pipeline: char n-grams → logistic regression
        lr_params = {"C": kwargs.pop("C", 1.0), "max_iter": kwargs.pop("max_iter", 1000)}
        lr_params.update(kwargs)

        self._ml_pipeline = SKLPipeline([
            ("vectorizer", CountVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=8000,
            )),
            ("classifier", LogisticRegression(**lr_params)),
        ])
        self._ml_pipeline.fit(filtered_texts, filtered_labels)

        logger.info(
            "SentimentDetector ML model trained on %d examples across %d classes.",
            len(filtered_texts),
            len(set(filtered_labels)),
        )
        return self

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, conversation: Dict) -> List[Signal]:
        turns = conversation.get("conversation", [])
        conv_id = conversation.get("conversation_id", "")

        if not turns:
            return []

        signals: List[Signal] = []

        for i, turn in enumerate(turns):
            role = turn.get("role", "")
            if role not in ("user", "human"):
                continue

            text = turn.get("content", "").strip()
            if not text or len(text) < 3:
                continue

            if self.require_adjacent_assistant:
                if i == 0 or turns[i - 1].get("role") not in ("assistant", "gpt", "model"):
                    continue

            assistant_text = ""
            if i > 0 and turns[i - 1].get("role") in ("assistant", "gpt", "model"):
                assistant_text = turns[i - 1].get("content", "")

            # Check for false-positive guards (skips the entire turn)
            if any(g.search(text) for g in FALSE_POSITIVE_GUARDS):
                continue

            # --- Stage 1: regex patterns ---
            signal = self._extract_regex(text, conv_id, i, assistant_text)
            if signal is not None:
                signals.append(signal)
                continue

            # --- Stage 2: ML fallback ---
            if self.use_ml and self._ml_pipeline is not None:
                ml_signal = self._extract_ml(text, conv_id, i, assistant_text)
                if ml_signal is not None:
                    signals.append(ml_signal)

        logger.debug(
            "SentimentDetector extracted %d signals from conv %s",
            len(signals),
            conv_id,
        )
        return signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_regex(
        self,
        text: str,
        conv_id: str,
        turn_index: int,
        assistant_text: str,
    ) -> Optional[Signal]:
        """Try regex patterns; return a Signal or None."""
        # Check in priority order: correction > negative > positive
        for pattern, weight in CORRECTION_PATTERNS:
            if pattern.search(text) and weight >= self.min_confidence:
                return Signal(
                    conversation_id=conv_id,
                    turn_index=turn_index,
                    signal_type="correction",
                    confidence=weight,
                    evidence=text[:500],
                    context={
                        "method": "regex",
                        "matched_pattern": pattern.pattern[:100],
                        "assistant_response": assistant_text[:500],
                    },
                )

        for pattern, weight in NEGATIVE_PATTERNS:
            if pattern.search(text) and weight >= self.min_confidence:
                return Signal(
                    conversation_id=conv_id,
                    turn_index=turn_index,
                    signal_type="negative",
                    confidence=weight,
                    evidence=text[:500],
                    context={
                        "method": "regex",
                        "matched_pattern": pattern.pattern[:100],
                        "assistant_response": assistant_text[:500],
                    },
                )

        for pattern, weight in POSITIVE_PATTERNS:
            if pattern.search(text) and weight >= self.min_confidence:
                return Signal(
                    conversation_id=conv_id,
                    turn_index=turn_index,
                    signal_type="positive",
                    confidence=weight,
                    evidence=text[:500],
                    context={
                        "method": "regex",
                        "matched_pattern": pattern.pattern[:100],
                        "assistant_response": assistant_text[:500],
                    },
                )

        return None

    def _extract_ml(
        self,
        text: str,
        conv_id: str,
        turn_index: int,
        assistant_text: str,
    ) -> Optional[Signal]:
        """Try ML classifier; return a Signal or None."""
        if self._ml_pipeline is None:
            return None

        try:
            proba = self._ml_pipeline.predict_proba([text])[0]
            pred = int(np.argmax(proba))
            confidence = float(proba[pred])
        except Exception:
            logger.debug("ML prediction failed for text: %s", text[:80])
            return None

        if confidence < self.ml_confidence_threshold:
            return None

        signal_type = self._ml_label_map.get(pred, "negative")

        return Signal(
            conversation_id=conv_id,
            turn_index=turn_index,
            signal_type=signal_type,
            confidence=round(confidence, 4),
            evidence=text[:500],
            context={
                "method": "ml",
                "ml_confidence": round(confidence, 4),
                "assistant_response": assistant_text[:500],
            },
        )
