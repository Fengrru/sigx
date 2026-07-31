"""
Pipeline module for orchestrating signal extraction, filtering, and conversion.

This module provides the main Pipeline class that chains together extractors,
filters, and converters to process conversation data end-to-end.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, overload

from .converters.preference import (
    CHOSEN_SUBSEQUENT,
    to_dpo,
    to_kto,
)
from .exceptions import (
    BenchmarkError,
    ConfigurationError,
    ConversationFormatError,
    ExtractionError,
    SigXError,
)
from .extractors.base import BaseExtractor
from .filters.quality import QualityGate
from .types import KTOExample, PreferencePair, Signal, generate_report

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline: extract -> filter -> convert.

    Conversations are processed as a stream: only turns belonging to
    conversations that produced raw signals are kept in memory, so
    generators (e.g. stream_wildchat) can be consumed without
    materializing the whole dataset.

    Conversations without a "conversation_id" are assigned a positional
    id ("0", "1", ...) consistently across extraction and conversion.

    Usage:
        pipeline = Pipeline(
            extractors=[RephraseDetector(), SentimentDetector()],
            quality_gate=QualityGate(min_confidence=0.6),
        )
        signals = pipeline.run(conversations)
        pairs = pipeline.to_dpo(conversations)
    """

    def __init__(
        self,
        extractors: List[BaseExtractor],
        quality_gate: Optional[QualityGate] = None,
        chosen_strategy: str = CHOSEN_SUBSEQUENT,
    ):
        self.extractors = extractors
        # Default gate matches QualityGate's own default (0.6) so there is
        # a single default confidence threshold across the library.
        self.quality_gate = quality_gate or QualityGate()
        self.chosen_strategy = chosen_strategy
        logger.info(
            "Pipeline initialized with %d extractors, quality_gate min_confidence=%.2f, "
            "chosen_strategy=%s",
            len(self.extractors),
            self.quality_gate.min_confidence,
            self.chosen_strategy,
        )

    def _run_impl(
        self,
        conversations: Iterable[Dict],
        collect_turns: bool = False,
    ) -> Tuple[List[Signal], Dict, Dict[str, List[Dict]]]:
        """Single-pass extraction over a conversation stream.

        Returns (filtered_signals, report_dict, conv_map). conv_map only
        contains conversations that produced raw signals (and only when
        collect_turns=True), keeping memory bounded for large streams.
        """
        raw_signals: List[Signal] = []
        conv_map: Dict[str, List[Dict]] = {}
        n_convos = 0
        n_turns = 0

        for i, conv in enumerate(conversations):
            if not isinstance(conv, dict):
                raise ConversationFormatError(
                    f"conversation {i} must be a dict, got {type(conv).__name__}"
                )
            if not conv.get("conversation_id"):
                # Assign a positional id consistently for extractors AND
                # converters (previously they used different defaults,
                # silently producing zero pairs).
                conv = dict(conv)
                conv["conversation_id"] = str(i)
            cid = conv["conversation_id"]

            turns = conv.get("conversation", [])
            n_convos += 1
            n_turns += len(turns)

            conv_signals: List[Signal] = []
            for extractor in self.extractors:
                try:
                    conv_signals.extend(extractor(conv))
                except SigXError:
                    raise
                except Exception as err:
                    raise ExtractionError(
                        f"Extractor '{extractor.name}' failed on conversation '{cid}': {err}"
                    ) from err

            if conv_signals:
                raw_signals.extend(conv_signals)
                if collect_turns:
                    conv_map[cid] = turns

        logger.info("Ran pipeline on %d conversations (%d total turns)", n_convos, n_turns)

        filtered, filter_report = self.quality_gate.filter_with_report(raw_signals)
        logger.info(
            "QualityGate: %d raw → %d filtered (dropped %d, retention %s)",
            filter_report["before"],
            filter_report["after"],
            filter_report["dropped"],
            filter_report["retention"],
        )

        report = dict(filter_report)
        report["conversations"] = n_convos
        report["total_turns"] = n_turns
        report["raw_signals"] = len(raw_signals)
        report["filtered_signals"] = len(filtered)

        return filtered, report, conv_map

    @overload
    def run(
        self,
        conversations: Iterable[Dict],
        return_report: Literal[False] = ...,
    ) -> List[Signal]: ...

    @overload
    def run(
        self,
        conversations: Iterable[Dict],
        return_report: Literal[True],
    ) -> Tuple[List[Signal], Dict]: ...

    def run(
        self,
        conversations: Iterable[Dict],
        return_report: bool = False,
    ) -> List[Signal] | Tuple[List[Signal], Dict]:
        """
        Run all extractors over conversations, then filter through quality gate.

        Args:
            conversations: Iterable of conversation dicts (a generator is
                fine; it is consumed in a single pass).
            return_report: If True, also return a dict with stats.

        Returns:
            Filtered signals, or (signals, report_dict) if return_report=True.
        """
        filtered, report, _ = self._run_impl(conversations)
        if return_report:
            return filtered, report
        return filtered

    @overload
    def to_dpo(
        self,
        conversations: Iterable[Dict],
        return_report: Literal[False] = ...,
    ) -> List[PreferencePair]: ...

    @overload
    def to_dpo(
        self,
        conversations: Iterable[Dict],
        return_report: Literal[True],
    ) -> Tuple[List[PreferencePair], Dict]: ...

    def to_dpo(
        self,
        conversations: Iterable[Dict],
        return_report: bool = False,
    ) -> List[PreferencePair] | Tuple[List[PreferencePair], Dict]:
        """
        End-to-end: extract signals, filter, convert to DPO pairs.
        """
        signals, report, conv_map = self._run_impl(conversations, collect_turns=True)

        pairs = to_dpo(signals, conv_map, chosen_strategy=self.chosen_strategy)

        if return_report:
            return pairs, report
        return pairs

    @overload
    def to_kto(
        self,
        conversations: Iterable[Dict],
        return_report: Literal[False] = ...,
    ) -> List[KTOExample]: ...

    @overload
    def to_kto(
        self,
        conversations: Iterable[Dict],
        return_report: Literal[True],
    ) -> Tuple[List[KTOExample], Dict]: ...

    def to_kto(
        self,
        conversations: Iterable[Dict],
        return_report: bool = False,
    ) -> List[KTOExample] | Tuple[List[KTOExample], Dict]:
        """
        End-to-end: extract signals, filter, convert to KTO examples.
        """
        signals, report, conv_map = self._run_impl(conversations, collect_turns=True)

        examples = to_kto(signals, conv_map)

        if return_report:
            return examples, report
        return examples

    def report(self, conversations: Iterable[Dict]) -> str:
        """Run the pipeline and return a text report."""
        filtered, run_report, _ = self._run_impl(conversations)
        report_obj = generate_report(
            filtered, run_report["conversations"], run_report["total_turns"]
        )

        lines = [
            report_obj.summary(),
            "",
            f"  Raw signals:    {run_report['before']}",
            f"  After filter:   {run_report['after']}",
            f"  Dropped:        {run_report['dropped']} ({run_report['retention']})",
        ]
        return "\n".join(lines)

    def evaluate(
        self,
        benchmark: List[Dict] | str,
    ) -> Dict:
        """
        Evaluate extraction quality against a labeled benchmark.

        Each benchmark item should have:
            - "conversation_id" (str)
            - "conversation" (list of turn dicts)
            - "ground_truth" (list of {"turn_index": int, "signal_type": str})

        Returns a dict with per-type and overall precision, recall, F1.
        """
        import json
        from pathlib import Path

        if isinstance(benchmark, str):
            path = Path(benchmark)
            if not path.exists():
                raise BenchmarkError(f"Benchmark file not found: {benchmark}")
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as err:
                raise BenchmarkError(f"Invalid JSON in benchmark file: {benchmark}") from err
            items = data.get("conversations", data)
        elif isinstance(benchmark, list):
            items = benchmark
        else:
            raise ConfigurationError("benchmark must be a list or file path string")

        if not items:
            return {"overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}}

        # 1. Collect ground truth
        gt_map: Dict[str, Dict[int, str]] = {}  # conv_id -> {turn_index: signal_type}
        gt_convos = []
        for item in items:
            cid = item.get("conversation_id", "")
            gt_map[cid] = {}
            for gt in item.get("ground_truth", []):
                gt_map[cid][gt["turn_index"]] = gt["signal_type"]
            gt_convos.append(
                {
                    "conversation_id": cid,
                    "conversation": item.get("conversation", []),
                }
            )

        # 2. Run pipeline to get predictions
        pred_signals = self.run(gt_convos)

        # Keep the highest-confidence prediction per (conversation, turn) so
        # the outcome does not depend on extractor ordering.
        pred_best: Dict[str, Dict[int, Tuple[str, float]]] = {}
        for sig in pred_signals:
            cid = sig.conversation_id
            if cid not in pred_best:
                pred_best[cid] = {}
            prev = pred_best[cid].get(sig.turn_index)
            if prev is None or sig.confidence > prev[1]:
                pred_best[cid][sig.turn_index] = (sig.signal_type, sig.confidence)

        pred_map: Dict[str, Dict[int, str]] = {
            cid: {turn: st for turn, (st, _conf) in turns.items()}
            for cid, turns in pred_best.items()
        }

        # 3. Match predictions against ground truth.
        # all_types is the union of ground-truth AND predicted types, so
        # spurious predictions of unlabeled types still count as false
        # positives instead of being silently ignored.
        gt_types = set(s["signal_type"] for item in items for s in item.get("ground_truth", []))
        pred_types = set(stype for turns in pred_map.values() for stype in turns.values())
        all_types = sorted(gt_types | pred_types)

        tp: Dict[str, int] = {t: 0 for t in all_types}
        fp: Dict[str, int] = {t: 0 for t in all_types}
        fn: Dict[str, int] = {t: 0 for t in all_types}

        for item in items:
            cid = item.get("conversation_id", "")
            gt_turns = gt_map.get(cid, {})
            pd_turns = pred_map.get(cid, {})

            all_turns = set(gt_turns.keys()) | set(pd_turns.keys())
            for turn_idx in all_turns:
                gt_type = gt_turns.get(turn_idx)
                pd_type = pd_turns.get(turn_idx)

                if gt_type is not None and pd_type is not None:
                    if gt_type == pd_type:
                        tp[gt_type] += 1
                    else:
                        fp[pd_type] += 1
                        fn[gt_type] += 1
                elif gt_type is not None:
                    fn[gt_type] += 1
                elif pd_type is not None:
                    fp[pd_type] += 1

        # 4. Calculate metrics per type
        def _f1(p: float, r: float) -> float:
            if p + r == 0:
                return 0.0
            return 2 * p * r / (p + r)

        per_type = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for stype in all_types:
            _tp = tp.get(stype, 0)
            _fp = fp.get(stype, 0)
            _fn = fn.get(stype, 0)
            support = _tp + _fn
            precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
            recall = _tp / support if support > 0 else 0.0
            per_type[stype] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(_f1(precision, recall), 4),
                "support": support,
            }
            total_tp += _tp
            total_fp += _fp
            total_fn += _fn

        # 5. Overall (micro-averaged across all ground truth signals)
        overall_support = total_tp + total_fn
        overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_r = total_tp / overall_support if overall_support > 0 else 0.0

        metrics: Dict[str, Any] = {
            "overall": {
                "precision": round(overall_p, 4),
                "recall": round(overall_r, 4),
                "f1": round(_f1(overall_p, overall_r), 4),
                "support": overall_support,
            },
            "per_type": per_type,
        }

        logger.info(
            "Evaluation: overall F1=%.4f (P=%.4f R=%.4f) on %d items",
            metrics["overall"]["f1"],
            metrics["overall"]["precision"],
            metrics["overall"]["recall"],
            len(items),
        )

        return metrics
