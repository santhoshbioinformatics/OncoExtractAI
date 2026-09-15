"""
Evaluation metrics for OncoExtractAI-QA.
Computes precision, recall, F1, and other metrics for extraction methods.
"""

import re
from typing import Dict, List, Optional, Any
from collections import defaultdict
from decimal import Decimal, InvalidOperation

from .schemas import (
    CORE_VARIABLES,
    ComparisonResult,
    DocumentationStatus,
    ExtractionResult,
    GoldAnnotation,
    OverallMetrics,
    VariableExtraction,
    VariableMetrics,
    evidence_text_supports_status,
    evidence_text_supports_value,
    local_evidence_context,
)


class Evaluator:
    """Computes evaluation metrics for extraction results."""

    def __init__(self):
        pass

    @staticmethod
    def _all_evidence_is_exact(var: VariableExtraction, report_text: str) -> bool:
        return bool(var.evidence) and all(
            0 <= evidence.start_offset < evidence.end_offset <= len(report_text)
            and report_text[evidence.start_offset:evidence.end_offset] == evidence.text
            for evidence in var.evidence
        )

    def _evidence_supports_value(self, var: VariableExtraction, report_text: str) -> bool:
        """Check exact anchoring plus a conservative value-to-evidence relationship."""

        if var.extracted_value is None or not self._all_evidence_is_exact(var, report_text):
            return False
        value = var.extracted_value
        for evidence in var.evidence or []:
            if not (
                0 <= evidence.start_offset < evidence.end_offset <= len(report_text)
                and report_text[evidence.start_offset:evidence.end_offset] == evidence.text
            ):
                continue
            if evidence_text_supports_value(
                var.variable_name,
                value,
                evidence.text,
                local_evidence_context(
                    report_text, evidence.start_offset, evidence.end_offset
                ),
            ):
                return True
        return False

    def _evidence_supports_status(
        self, var: VariableExtraction, report_text: str
    ) -> bool:
        """Require exact spans whose local wording supports the claimed status."""

        if not var.evidence or not self._all_evidence_is_exact(var, report_text):
            return var.documentation_status == DocumentationStatus.NOT_DOCUMENTED
        evidence_texts = [span.text for span in var.evidence]
        contexts = [
            local_evidence_context(report_text, span.start_offset, span.end_offset)
            for span in var.evidence
        ]
        return evidence_text_supports_status(
            var.variable_name,
            var.documentation_status,
            evidence_texts,
            value=var.extracted_value,
            alternatives=getattr(var, "alternatives", None),
            context_texts=contexts,
        )

    @staticmethod
    def _validate_cohort(
        extractions: List[ExtractionResult],
        gold_annotations: List[GoldAnnotation],
        report_texts: Dict[str, str],
        expected_method: str | None = None,
    ) -> None:
        if not extractions:
            raise ValueError("Evaluation requires at least one extraction result.")

        extraction_ids = [item.report_id for item in extractions]
        if len(extraction_ids) != len(set(extraction_ids)):
            raise ValueError("Evaluation extraction report IDs must be unique.")
        extraction_id_set = set(extraction_ids)
        if extraction_id_set != set(report_texts):
            raise ValueError("Extraction and report-text cohorts must contain the same report IDs.")

        methods = {item.method for item in extractions}
        if len(methods) != 1:
            raise ValueError("Evaluation cannot mix extraction methods in one cohort.")
        if expected_method is not None and methods != {expected_method}:
            raise ValueError(f"Expected only {expected_method} extraction results.")

        gold_keys = [(item.report_id, item.variable_name) for item in gold_annotations]
        if len(gold_keys) != len(set(gold_keys)):
            raise ValueError("Gold annotations must be unique by report and variable.")
        expected_keys = {
            (report_id, variable_name)
            for report_id in extraction_id_set
            for variable_name in CORE_VARIABLES
        }
        if set(gold_keys) != expected_keys:
            raise ValueError(
                "Gold annotations must contain exactly four core fields for every evaluated report."
            )

        extraction_by_id = {item.report_id: item for item in extractions}
        gold_cancer_types: dict[str, set[str]] = defaultdict(set)
        for annotation in gold_annotations:
            if annotation.cancer_type is not None:
                gold_cancer_types[annotation.report_id].add(annotation.cancer_type)

            if annotation.evidence_text is None:
                continue
            assert annotation.evidence_start_offset is not None
            assert annotation.evidence_end_offset is not None
            report_text = report_texts[annotation.report_id]
            if not (
                0 <= annotation.evidence_start_offset
                < annotation.evidence_end_offset
                <= len(report_text)
                and report_text[
                    annotation.evidence_start_offset : annotation.evidence_end_offset
                ]
                == annotation.evidence_text
            ):
                raise ValueError(
                    "Gold evidence offsets must resolve to the exact report substring."
                )

            context = local_evidence_context(
                report_text,
                annotation.evidence_start_offset,
                annotation.evidence_end_offset,
            )
            if not evidence_text_supports_status(
                annotation.variable_name,
                annotation.documentation_status,
                [annotation.evidence_text],
                value=annotation.gold_value,
                context_texts=[context],
            ):
                raise ValueError(
                    "Gold evidence does not support its documentation status in context."
                )

        for report_id, cancer_types in gold_cancer_types.items():
            if len(cancer_types) > 1:
                raise ValueError(
                    f"Gold annotations disagree on cancer_type for {report_id}."
                )
            extraction_type = extraction_by_id[report_id].cancer_type
            gold_type = next(iter(cancer_types), None)
            if extraction_type is not None and gold_type is not None and extraction_type != gold_type:
                raise ValueError(
                    f"Extraction and gold cancer_type disagree for {report_id}."
                )

    def _result_schema_valid(self, extraction: ExtractionResult, report_text: str) -> bool:
        try:
            validated = ExtractionResult.model_validate(extraction.model_dump(mode="python"))
        except Exception:
            return False
        names = [variable.variable_name for variable in validated.variables]
        if len(names) != len(CORE_VARIABLES) or set(names) != set(CORE_VARIABLES):
            return False
        for variable in validated.variables:
            if variable.evidence and not self._all_evidence_is_exact(variable, report_text):
                return False
            if variable.extracted_value is not None and not variable.evidence:
                return False
        return True

    def evaluate_single_report(
        self,
        extraction: ExtractionResult,
        gold_annotations: List[GoldAnnotation],
        report_text: str
    ) -> Dict[str, Any]:
        """Evaluate extraction results for a single report against gold annotations."""
        matching_gold = [
            annotation
            for annotation in gold_annotations
            if annotation.report_id == extraction.report_id
        ]
        self._validate_cohort(
            [extraction], matching_gold, {extraction.report_id: report_text}
        )
        results = {
            'report_id': extraction.report_id,
            'variables': {},
            'qa_issues_detected': len(extraction.qa_issues),
            'manual_review_flagged': extraction.manual_review_required,
        }

        # Build gold annotation lookup
        gold_lookup = {}
        for ga in matching_gold:
            gold_lookup[ga.variable_name] = ga

        for var in extraction.variables:
            var_results = {
                'extracted_value': var.extracted_value,
                'documentation_status': var.documentation_status.value,
                'has_evidence': var.evidence is not None and len(var.evidence) > 0,
                'evidence_valid': None,
                'value_correct': False,
                'status_correct': False,
            }

            if var.variable_name in gold_lookup:
                gold = gold_lookup[var.variable_name]

                # Check value correctness
                if gold.gold_value is None and var.extracted_value is None:
                    var_results['value_correct'] = True  # Both abstained
                elif gold.gold_value is not None and var.extracted_value is not None:
                    var_results['value_correct'] = self._values_match(var.extracted_value, gold.gold_value)

                # Check status correctness
                var_results['status_correct'] = (var.documentation_status == gold.documentation_status)

                # Check evidence validity
                if var.evidence and (
                    gold.evidence_text is None
                    or gold.evidence_start_offset is None
                    or gold.evidence_end_offset is None
                ):
                    var_results['evidence_valid'] = False
                elif (
                    var.evidence
                    and gold.evidence_text
                    and gold.evidence_start_offset is not None
                    and gold.evidence_end_offset is not None
                ):
                    gold_start = gold.evidence_start_offset
                    gold_end = gold.evidence_end_offset
                    spans_align_without_partial_overlap = all(
                        (
                            gold_start <= ev.start_offset
                            and ev.end_offset <= gold_end
                        )
                        or (
                            ev.start_offset <= gold_start
                            and gold_end <= ev.end_offset
                        )
                        for ev in var.evidence
                    )
                    var_results['evidence_valid'] = (
                        spans_align_without_partial_overlap
                        and self._evidence_supports_status(var, report_text)
                    )

            results['variables'][var.variable_name] = var_results

        return results

    def _values_match(self, extracted: Optional[str], gold: Optional[str]) -> bool:
        """Check if extracted value matches gold value with some flexibility."""
        if extracted is None or gold is None:
            return extracted is None and gold is None
        if extracted == gold:
            return True

        # Normalize for comparison
        extracted_norm = re.sub(r'[^a-z0-9]', '', extracted.lower())
        gold_norm = re.sub(r'[^a-z0-9]', '', gold.lower())

        if extracted_norm == gold_norm:
            return True

        # Canonical centimetre values compare exactly after decimal normalization.
        if 'cm' in extracted.lower() and 'cm' in gold.lower():
            ext_num = re.search(r'(\d+\.?\d*)', extracted)
            gold_num = re.search(r'(\d+\.?\d*)', gold)
            if ext_num and gold_num:
                try:
                    return Decimal(ext_num.group(1)) == Decimal(gold_num.group(1))
                except InvalidOperation:
                    return False

        return False

    def compute_overall_metrics(
        self,
        extractions: List[ExtractionResult],
        gold_annotations: List[GoldAnnotation],
        report_texts: Dict[str, str]
    ) -> OverallMetrics:
        """Compute overall metrics across all reports."""
        self._validate_cohort(extractions, gold_annotations, report_texts)
        # Build gold annotation lookup by report
        gold_by_report = defaultdict(list)
        for ga in gold_annotations:
            gold_by_report[ga.report_id].append(ga)

        # Aggregate counts
        tp = defaultdict(int)  # True positives per variable
        fp = defaultdict(int)  # False positives per variable
        fn = defaultdict(int)  # False negatives per variable
        tn = defaultdict(int)  # True negatives per variable
        unsupported = defaultdict(int)
        abstained = defaultdict(int)
        evidence_correct = defaultdict(int)
        evidence_total = defaultdict(int)
        status_correct = defaultdict(int)
        status_total = defaultdict(int)

        total_reports = len(extractions)
        total_manual_review = 0
        total_schema_valid = 0
        expected_abstentions = 0
        appropriate_abstentions = 0
        undocumented_gold_fields = 0
        undocumented_false_positives = 0
        referral_tp = 0
        referral_fp = 0
        referral_fn = 0
        gold_review_by_report = {
            report_id: any(item.manual_review_required for item in items)
            for report_id, items in gold_by_report.items()
        }

        for extraction in extractions:
            report_id = extraction.report_id
            gold_list = gold_by_report.get(report_id, [])
            report_text = report_texts.get(report_id, '')

            # A valid result must round-trip, contain all four fields exactly once,
            # and keep every returned value anchored to exact report offsets.
            if self._result_schema_valid(extraction, report_text):
                total_schema_valid += 1

            # Check manual review
            if extraction.manual_review_required:
                total_manual_review += 1
            expected_review = gold_review_by_report[report_id]
            if extraction.manual_review_required and expected_review:
                referral_tp += 1
            elif extraction.manual_review_required and not expected_review:
                referral_fp += 1
            elif not extraction.manual_review_required and expected_review:
                referral_fn += 1

            # Evaluate each variable
            for var in extraction.variables:
                var_name = var.variable_name
                gold = next((g for g in gold_list if g.variable_name == var_name), None)

                if not gold:
                    continue

                status_total[var_name] += 1
                if var.documentation_status == gold.documentation_status:
                    status_correct[var_name] += 1

                # Count abstentions
                if var.extracted_value is None:
                    if gold.gold_value is None:
                        tn[var_name] += 1  # Correct abstention
                    else:
                        fn[var_name] += 1  # Should have extracted
                    abstained[var_name] += 1
                else:
                    # Value was extracted
                    evidence_total[var_name] += 1
                    if self._evidence_supports_value(var, report_text):
                        evidence_correct[var_name] += 1
                    else:
                        unsupported[var_name] += 1

                    # Check value correctness
                    if gold.gold_value is None:
                        # Extracted a value when gold says abstain / not assignable
                        fp[var_name] += 1
                    elif self._values_match(var.extracted_value, gold.gold_value):
                        # Count as TP when value matches; status nuances tracked separately
                        tp[var_name] += 1
                    else:
                        fp[var_name] += 1
                        fn[var_name] += 1

                if gold.gold_value is None:
                    expected_abstentions += 1
                    if var.extracted_value is None:
                        appropriate_abstentions += 1
                if gold.documentation_status == DocumentationStatus.NOT_DOCUMENTED:
                    undocumented_gold_fields += 1
                    if var.extracted_value is not None:
                        undocumented_false_positives += 1

        # Compute metrics per variable
        variable_metrics = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for var_name in CORE_VARIABLES:
            v_tp = tp[var_name]
            v_fp = fp[var_name]
            v_fn = fn[var_name]

            precision = v_tp / (v_tp + v_fp) if (v_tp + v_fp) > 0 else None
            recall = v_tp / (v_tp + v_fn) if (v_tp + v_fn) > 0 else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and (precision + recall) > 0
                else 0.0 if precision is not None and recall is not None
                else None
            )

            variable_metrics.append(VariableMetrics(
                variable_name=var_name,
                precision=round(precision, 4) if precision is not None else None,
                recall=round(recall, 4) if recall is not None else None,
                f1_score=round(f1, 4) if f1 is not None else None,
                true_positives=v_tp,
                false_positives=v_fp,
                false_negatives=v_fn,
                true_negatives=tn[var_name],
                unsupported_count=unsupported[var_name],
                abstained_count=abstained[var_name],
                evidence_correct_count=evidence_correct[var_name],
                evidence_total_count=evidence_total[var_name],
                status_correct_count=status_correct[var_name],
                status_total_count=status_total[var_name],
            ))

            total_tp += v_tp
            total_fp += v_fp
            total_fn += v_fn

        # Overall metrics
        total_non_abstained = sum(vm.evidence_total_count for vm in variable_metrics)
        total_unsupported = sum(vm.unsupported_count for vm in variable_metrics)
        unsupported_rate = total_unsupported / total_non_abstained if total_non_abstained > 0 else None

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if overall_precision is not None and overall_recall is not None
            and (overall_precision + overall_recall) > 0
            else 0.0 if overall_precision is not None and overall_recall is not None
            else None
        )

        # Coverage: proportion of variables that were extracted (not abstained)
        total_variables = len(gold_annotations)
        total_abstained = sum(vm.abstained_count for vm in variable_metrics)
        coverage = (total_variables - total_abstained) / total_variables if total_variables > 0 else 0.0

        appropriate_abstention = (
            appropriate_abstentions / expected_abstentions
            if expected_abstentions > 0
            else None
        )

        total_status_correct = sum(vm.status_correct_count for vm in variable_metrics)
        total_evidence_correct = sum(vm.evidence_correct_count for vm in variable_metrics)
        status_accuracy = (
            total_status_correct / total_variables if total_variables > 0 else 0.0
        )
        exact_evidence_rate = (
            total_evidence_correct / total_non_abstained
            if total_non_abstained > 0
            else None
        )

        referral_precision = (
            referral_tp / (referral_tp + referral_fp)
            if referral_tp + referral_fp > 0
            else None
        )
        referral_recall = (
            referral_tp / (referral_tp + referral_fn)
            if referral_tp + referral_fn > 0
            else None
        )

        method = extractions[0].method

        return OverallMetrics(
            method=method,
            report_count=total_reports,
            fields_evaluated=total_variables,
            returned_value_count=total_non_abstained,
            gold_value_count=total_tp + total_fn,
            expected_abstention_count=expected_abstentions,
            appropriate_abstention_count=appropriate_abstentions,
            undocumented_gold_field_count=undocumented_gold_fields,
            undocumented_false_positive_count=undocumented_false_positives,
            manual_review_count=total_manual_review,
            schema_valid_count=total_schema_valid,
            unsupported_extraction_rate=(
                round(unsupported_rate, 4) if unsupported_rate is not None else None
            ),
            overall_precision=(round(overall_precision, 4) if overall_precision is not None else None),
            overall_recall=(round(overall_recall, 4) if overall_recall is not None else None),
            overall_f1=round(overall_f1, 4) if overall_f1 is not None else None,
            appropriate_abstention_rate=(
                round(appropriate_abstention, 4) if appropriate_abstention is not None else None
            ),
            false_positive_rate_undocumented=(
                round(undocumented_false_positives / undocumented_gold_fields, 4)
                if undocumented_gold_fields > 0
                else None
            ),
            coverage=round(coverage, 4),
            manual_review_referral_rate=round(total_manual_review / total_reports, 4) if total_reports > 0 else 0.0,
            schema_validity_rate=round(total_schema_valid / total_reports, 4) if total_reports > 0 else 0.0,
            documentation_status_accuracy=round(status_accuracy, 4),
            documentation_status_correct_count=total_status_correct,
            exact_evidence_rate=(
                round(exact_evidence_rate, 4) if exact_evidence_rate is not None else None
            ),
            exact_evidence_correct_count=total_evidence_correct,
            exact_evidence_total_count=total_non_abstained,
            gold_manual_review_report_count=sum(gold_review_by_report.values()),
            review_referral_true_positive_count=referral_tp,
            review_referral_false_positive_count=referral_fp,
            review_referral_false_negative_count=referral_fn,
            manual_review_referral_precision=(
                round(referral_precision, 4) if referral_precision is not None else None
            ),
            manual_review_referral_recall=(
                round(referral_recall, 4) if referral_recall is not None else None
            ),
            variable_metrics=variable_metrics,
        )

    def compare_methods(
        self,
        baseline_extractions: List[ExtractionResult],
        evidence_first_extractions: List[ExtractionResult],
        gold_annotations: List[GoldAnnotation],
        report_texts: Dict[str, str]
    ) -> ComparisonResult:
        """Compare baseline and evidence-first methods."""
        self._validate_cohort(
            baseline_extractions,
            gold_annotations,
            report_texts,
            expected_method="baseline",
        )
        self._validate_cohort(
            evidence_first_extractions,
            gold_annotations,
            report_texts,
            expected_method="evidence_first",
        )
        if {item.report_id for item in baseline_extractions} != {
            item.report_id for item in evidence_first_extractions
        }:
            raise ValueError("Baseline and evidence-first cohorts must contain the same reports.")
        baseline_metrics = self.compute_overall_metrics(
            baseline_extractions, gold_annotations, report_texts
        )
        evidence_first_metrics = self.compute_overall_metrics(
            evidence_first_extractions, gold_annotations, report_texts
        )

        return ComparisonResult(
            baseline_metrics=baseline_metrics,
            evidence_first_metrics=evidence_first_metrics,
            unsupported_rate_reduction=(
                round(
                    baseline_metrics.unsupported_extraction_rate
                    - evidence_first_metrics.unsupported_extraction_rate,
                    4,
                )
                if baseline_metrics.unsupported_extraction_rate is not None
                and evidence_first_metrics.unsupported_extraction_rate is not None
                else None
            ),
            precision_improvement=(
                round(
                    evidence_first_metrics.overall_precision
                    - baseline_metrics.overall_precision,
                    4,
                )
                if baseline_metrics.overall_precision is not None
                and evidence_first_metrics.overall_precision is not None
                else None
            ),
            recall_change=(
                round(
                    evidence_first_metrics.overall_recall
                    - baseline_metrics.overall_recall,
                    4,
                )
                if baseline_metrics.overall_recall is not None
                and evidence_first_metrics.overall_recall is not None
                else None
            ),
            coverage_change=round(
                evidence_first_metrics.coverage - baseline_metrics.coverage, 4
            ),
        )
