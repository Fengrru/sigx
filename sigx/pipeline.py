"""
Pipeline module for orchestrating signal extraction, filtering, and conversion.

This module provides the main Pipeline class that chains together extractors,
filters, and converters to process conversation data end-to-end.
"""

import logging
from typing import Dict, Iterable, List, Optional

from .converters.preference import (
    CHOSEN_SUBSEQUENT,
    to_dpo,
    to_kto,
)
from .extractors.base import BaseExtractor
from .filters.quality import QualityGate
from .types import KTOExample, PreferencePair, Signal, generate_report

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline: extract -> filter -> convert.

    Usage:
        pipeline = Pipeline(
            extractors=[RephraseDetector(), SentimentDetector()],
            quality_gate=QualityGate(min_confidence=0.6),
        )
        signals = pipeline.run(conversations)
        pairs = pipeline.to_dpo(signals)
    """

    def __init__(
        self,
        extractors: List[BaseExtractor],
        quality_gate: Optional[QualityGate] = None,
        chosen_strategy: str = CHOSEN_SUBSEQUENT,
    ):
        self.extractors = extractors
        self.quality_gate = quality_gate or QualityGate(min_confidence=0.5)
        self.chosen_strategy = chosen_strategy
        logger.info(
            "Pipeline initialized with %d extractors, quality_gate min_confidence=%.2f, "
            "chosen_strategy=%s",
            len(self.extractors),
            self.quality_gate.min_confidence,
            self.chosen_strategy,
        )

    def run(
        self,
        conversations: Iterable[Dict],
        return_report: bool = False,
    ) -> List[Signal] | tuple[List[Signal], Dict]:
        """
        Run all extractors over conversations, then filter through quality gate.

        Args:
            conversations: Iterable of conversation dicts.
            return_report: If True, also return a dict with stats.

        Returns:
            Filtered signals, or (signals, report_dict) if return_report=True.
        """
        convos = list(conversations)
        n_convos = len(convos)
        n_turns = sum(len(c.get("conversation", [])) for c in convos)
        logger.info("Running pipeline on %d conversations (%d total turns)", n_convos, n_turns)

        raw_signals: List[Signal] = []
        for conv in convos:
            for extractor in self.extractors:
                raw_signals.extend(extractor(conv))

        filtered, filter_report = self.quality_gate.filter_with_report(raw_signals)
        logger.info(
            "QualityGate: %d raw → %d filtered (dropped %d, retention %s)",
            filter_report["before"],
            filter_report["after"],
            filter_report["dropped"],
            filter_report["retention"],
        )

        if return_report:
            report = filter_report.copy()
            report["conversations"] = n_convos
            report["total_turns"] = n_turns
            report["raw_signals"] = len(raw_signals)
            report["filtered_signals"] = len(filtered)
            return filtered, report

        return filtered

    def to_dpo(
        self,
        conversations: Iterable[Dict],
        return_report: bool = False,
    ) -> List[PreferencePair] | tuple[List[PreferencePair], Dict]:
        """
        End-to-end: extract signals, filter, convert to DPO pairs.

        Args:
            conversations: Iterable of conversation dicts.
            return_report: If True, also return stats.

        Returns:
            DPO preference pairs.
        """
        convos = list(conversations)
        signals, report = self.run(convos, return_report=True)

        conv_map = {
            c.get("conversation_id", str(i)): c.get("conversation", [])
            for i, c in enumerate(convos)
        }

        pairs = to_dpo(signals, conv_map, chosen_strategy=self.chosen_strategy)

        if return_report:
            return pairs, report
        return pairs

    def to_kto(
        self,
        conversations: Iterable[Dict],
        return_report: bool = False,
    ) -> List[KTOExample] | tuple[List[KTOExample], Dict]:
        """
        End-to-end: extract signals, filter, convert to KTO examples.
        """
        convos = list(conversations)
        signals, report = self.run(convos, return_report=True)

        conv_map = {
            c.get("conversation_id", str(i)): c.get("conversation", [])
            for i, c in enumerate(convos)
        }

        examples = to_kto(signals, conv_map)

        if return_report:
            return examples, report
        return examples

    def report(self, conversations: Iterable[Dict]) -> str:
        """Run the pipeline and return a text report."""
        convos = list(conversations)
        n_convos = len(convos)
        n_turns = sum(len(c.get("conversation", [])) for c in convos)

        raw_signals = []
        for conv in convos:
            for extractor in self.extractors:
                raw_signals.extend(extractor(conv))

        _, filter_report = self.quality_gate.filter_with_report(raw_signals)
        filtered = self.quality_gate(raw_signals)
        report_obj = generate_report(filtered, n_convos, n_turns)

        lines = [
            report_obj.summary(),
            "",
            f"  Raw signals:    {filter_report['before']}",
            f"  After filter:   {filter_report['after']}",
            f"  Dropped:        {filter_report['dropped']} ({filter_report['retention']})",
        ]
        return "\n".join(lines)

    def evaluate(
        self,
        benchmark: List[Dict],
    ) -> Dict:
        """
        Evaluate extraction quality against a labeled benchmark.

        Each benchmark item should have:
            - "conversation_id" (str)
            - "conversation" (list of turn dicts)
            - "ground_truth" (list of {"turn_index": int, "signal_type": str})

        Returns a dict with per-type and overall precision, recall, F1.

        Example:
            >>> benchmark = [
            ...     {
            ...         "conversation_id": "1",
            ...         "conversation": [...],
            ...         "ground_truth": [{"turn_index": 2, "signal_type": "negative"}],
            ...     }
            ... ]
            >>> metrics = pipeline.evaluate(benchmark)
            >>> print(metrics["overall"]["f1"])
        """
        import json
        from pathlib import Path

        # Support passing a file path string
        if isinstance(benchmark, str):
            path = Path(benchmark)
            if not path.exists():
                raise FileNotFoundError(f"Benchmark file not found: {benchmark}")
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("conversations", data)
        elif isinstance(benchmark, list):
            items = benchmark
        else:
            raise TypeError("benchmark must be a list or file path string")

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
            gt_convos.append({
                "conversation_id": cid,
                "conversation": item.get("conversation", []),
            })

        # 2. Run pipeline to get predictions
        pred_signals = self.run(gt_convos)

        pred_map: Dict[str, Dict[int, str]] = {}
        for sig in pred_signals:
            cid = sig.conversation_id
            if cid not in pred_map:
                pred_map[cid] = {}
            pred_map[cid][sig.turn_index] = sig.signal_type

        # 3. Match predictions against ground truth
        all_types = sorted(set(
            s["signal_type"]
            for item in items
            for s in item.get("ground_truth", [])
        ))

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
                        if gt_type in tp:
                            tp[gt_type] += 1
                    else:
                        if pd_type in fp:
                            fp[pd_type] += 1
                        if gt_type in fn:
                            fn[gt_type] += 1
                elif gt_type is not None:
                    if gt_type in fn:
                        fn[gt_type] += 1
                elif pd_type is not None:
                    if pd_type in fp:
                        fp[pd_type] += 1

        # 4. Calculate metrics per type
        def _f1(p, r):
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

        result = {
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
            result["overall"]["f1"],
            result["overall"]["precision"],
            result["overall"]["recall"],
            len(items),
        )

        return result
