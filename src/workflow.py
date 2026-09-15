"""Application orchestration for extraction, validation, and QA."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import ValidationError

from .evidence_validator import EvidenceValidator
from .extractor import baseline_extract, evidence_first_extract
from .ml_model import MLModelError, ml_extract
from .qa_detector import QADetector
from .schemas import CORE_VARIABLES, ExtractionResult


ExtractionMethod = Literal["baseline", "evidence_first", "ml"]


class ExtractionPipelineError(RuntimeError):
    """A safe reviewer-facing extraction failure."""


def validate_result_contract(result: ExtractionResult) -> ExtractionResult:
    """Round-trip model output and enforce one result per core variable."""

    try:
        validated = ExtractionResult.model_validate(result.model_dump(mode="python"))
    except ValidationError as error:
        raise ExtractionPipelineError("Extraction output did not match the required schema.") from error

    names = [variable.variable_name for variable in validated.variables]
    missing = [name for name in CORE_VARIABLES if name not in names]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    extras = [name for name in names if name not in CORE_VARIABLES]
    if missing or duplicates or extras:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if duplicates:
            details.append(f"duplicated: {', '.join(duplicates)}")
        if extras:
            details.append(f"unsupported: {', '.join(extras)}")
        raise ExtractionPipelineError(
            "Extraction output had an invalid field set (" + "; ".join(details) + ")."
        )
    return validated


def run_extraction_pipeline(
    report: Mapping[str, Any],
    method: ExtractionMethod,
) -> ExtractionResult:
    """Run one extraction method and return schema-validated output."""

    report_id = str(report.get("report_id") or "LOCAL-REPORT")
    report_text = report.get("text")
    if not isinstance(report_text, str) or not report_text.strip():
        raise ExtractionPipelineError("Load non-empty report text before running extraction.")

    try:
        if method == "baseline":
            result = baseline_extract(report_text, report_id)
            result = EvidenceValidator().validate_and_flag(result, report_text)
        elif method == "evidence_first":
            result = evidence_first_extract(report_text, report_id)
            result = EvidenceValidator().validate_and_flag(result, report_text)
            result = QADetector().run_full_qa(report_text, result)
        elif method == "ml":
            result = ml_extract(report_text, report_id)
            result = EvidenceValidator().validate_and_flag(result, report_text)
            result = QADetector().run_full_qa(report_text, result)
        else:
            raise ExtractionPipelineError(f"Unsupported extraction method: {method}.")

        cancer_type = report.get("cancer_type")
        result.cancer_type = str(cancer_type) if cancer_type else None
        return validate_result_contract(result)
    except ExtractionPipelineError:
        raise
    except MLModelError as error:
        raise ExtractionPipelineError(str(error)) from error
    except (ValidationError, TypeError, ValueError) as error:
        raise ExtractionPipelineError(
            "Extraction failed safely because the report or structured output was malformed."
        ) from error


def run_both_pipelines(report: Mapping[str, Any]) -> dict[str, ExtractionResult]:
    return {
        "baseline": run_extraction_pipeline(report, "baseline"),
        "evidence_first": run_extraction_pipeline(report, "evidence_first"),
        "ml": run_extraction_pipeline(report, "ml"),
    }

