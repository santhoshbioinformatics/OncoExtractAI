"""Local sklearn ML extractor for lung pathology abstraction.

This module trains and runs a real machine-learning model over report text.
It is intentionally hybrid:

1. TF-IDF + logistic regression predict field values / documentation states
2. Exact substring anchoring attaches evidence spans from the source report
3. Clinical guardrails refuse invented pT/pN categories and unanchored values

The model is trained from the synthetic gold annotations shipped with the
prototype. It is a research demonstration model, not a clinical system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .extractor import (
    PATTERNS,
    _choose_best_match,
    _documented_no_value,
    _find_all_matches,
    _has_uncertainty,
    _issue_priority,
    _local_context,
    _make_evidence,
    _negation_evidence,
    _normalize_value,
    _resolve_current_matches,
)
from .schemas import (
    CORE_VARIABLES,
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    QAIssue,
    QAIssueType,
    VariableExtraction,
)
from .utils import DATA_DIR, PROJECT_ROOT, get_iso_timestamp, load_gold_annotations, load_synthetic_reports

ABSENT_LABEL = "__ABSENT__"
CANNOT_ASSIGN_LABEL = "__CANNOT_ASSIGN__"
CONFLICT_LABEL = "__CONFLICT__"
NEGATED_LABEL = "__NEGATED__"
UNCERTAIN_PREFIX = "__UNCERTAIN__::"

SPECIAL_LABELS = {
    ABSENT_LABEL,
    CANNOT_ASSIGN_LABEL,
    CONFLICT_LABEL,
    NEGATED_LABEL,
}

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pathology_ml_extractor.joblib"
# Absolute multiclass probabilities stay modest on the tiny synthetic set.
# Use a soft floor plus top-1 vs top-2 margin; exact anchoring remains mandatory.
MIN_CONFIDENCE = 0.08
MIN_MARGIN = 0.01


class MLModelError(RuntimeError):
    """Raised when the ML model cannot be trained, loaded, or applied safely."""


@dataclass(frozen=True)
class FieldPrediction:
    variable_name: str
    label: str
    confidence: float
    probabilities: dict[str, float]


def _require_sklearn():
    try:
        import joblib  # noqa: F401
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.pipeline import Pipeline  # noqa: F401
    except ImportError as error:
        raise MLModelError(
            "scikit-learn and joblib are required for the ML extractor. "
            "Install them with: pip install scikit-learn joblib"
        ) from error


def _status_label(status: DocumentationStatus, gold_value: str | None) -> str:
    value = (gold_value or "").strip()
    if status == DocumentationStatus.CONFLICTING:
        return CONFLICT_LABEL
    if status == DocumentationStatus.NEGATED:
        return NEGATED_LABEL
    if status == DocumentationStatus.NOT_DOCUMENTED and not value:
        return ABSENT_LABEL
    if status == DocumentationStatus.CANNOT_BE_ASSIGNED and not value:
        return CANNOT_ASSIGN_LABEL
    if not value:
        return ABSENT_LABEL
    if status == DocumentationStatus.UNCERTAIN:
        return f"{UNCERTAIN_PREFIX}{value}"
    return value


def _decode_label(label: str) -> tuple[str | None, DocumentationStatus | None]:
    if label == ABSENT_LABEL:
        return None, DocumentationStatus.NOT_DOCUMENTED
    if label == CANNOT_ASSIGN_LABEL:
        return None, DocumentationStatus.CANNOT_BE_ASSIGNED
    if label == CONFLICT_LABEL:
        return None, DocumentationStatus.CONFLICTING
    if label == NEGATED_LABEL:
        return None, DocumentationStatus.NEGATED
    if label.startswith(UNCERTAIN_PREFIX):
        return label[len(UNCERTAIN_PREFIX):], DocumentationStatus.UNCERTAIN
    return label, None


def _build_training_rows(
    reports: list[dict[str, Any]] | None = None,
    gold_rows: list[dict[str, Any]] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    reports = reports if reports is not None else load_synthetic_reports()
    gold_rows = gold_rows if gold_rows is not None else load_gold_annotations()
    texts = {
        str(report["report_id"]): str(report["text"])
        for report in reports
        if isinstance(report.get("report_id"), str) and isinstance(report.get("text"), str)
    }
    by_variable: dict[str, list[tuple[str, str]]] = {name: [] for name in CORE_VARIABLES}
    for row in gold_rows:
        report_id = str(row.get("report_id", ""))
        variable_name = str(row.get("variable_name", ""))
        if report_id not in texts or variable_name not in by_variable:
            continue
        status = DocumentationStatus(str(row.get("documentation_status", "not_documented")))
        gold_value = row.get("gold_value")
        gold_value = str(gold_value).strip() if gold_value not in (None, "") else None
        label = _status_label(status, gold_value)
        by_variable[variable_name].append((texts[report_id], label))
    for variable_name, rows in by_variable.items():
        if len(rows) < 2:
            raise MLModelError(
                f"Not enough labeled examples to train ML model for {variable_name}."
            )
        labels = {label for _, label in rows}
        if len(labels) < 2:
            # Single-class fields still train, but sklearn needs >=1 class.
            # Duplicate a tiny synthetic contrast only for pipeline fit stability.
            by_variable[variable_name] = rows + [(rows[0][0], ABSENT_LABEL)]
    return by_variable


def _make_pipeline(seed: int = 13):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


class PathologyMLExtractor:
    """Trainable multi-field pathology abstraction model."""

    def __init__(self, pipelines: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None):
        self.pipelines = pipelines or {}
        self.metadata = metadata or {
            "model_name": "pathology_tfidf_logreg_v1",
            "framework": "scikit-learn",
            "variables": list(CORE_VARIABLES),
        }

    @property
    def is_ready(self) -> bool:
        return all(name in self.pipelines for name in CORE_VARIABLES)

    def train(
        self,
        reports: list[dict[str, Any]] | None = None,
        gold_rows: list[dict[str, Any]] | None = None,
        seed: int = 13,
    ) -> "PathologyMLExtractor":
        _require_sklearn()
        training = _build_training_rows(reports=reports, gold_rows=gold_rows)
        pipelines: dict[str, Any] = {}
        label_counts: dict[str, dict[str, int]] = {}
        for variable_name, rows in training.items():
            texts = [text for text, _ in rows]
            labels = [label for _, label in rows]
            pipeline = _make_pipeline(seed=seed)
            pipeline.fit(texts, labels)
            pipelines[variable_name] = pipeline
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            label_counts[variable_name] = counts
        self.pipelines = pipelines
        self.metadata = {
            **self.metadata,
            "trained": True,
            "n_reports": len({report["report_id"] for report in (reports or load_synthetic_reports())}),
            "label_counts": label_counts,
            "min_confidence": MIN_CONFIDENCE,
        }
        return self

    def save(self, path: str | Path | None = None) -> Path:
        _require_sklearn()
        import joblib

        if not self.is_ready:
            raise MLModelError("Cannot save an untrained ML extractor.")
        target = Path(path) if path is not None else DEFAULT_MODEL_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipelines": self.pipelines, "metadata": self.metadata}, target)
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PathologyMLExtractor":
        _require_sklearn()
        import joblib

        target = Path(path) if path is not None else DEFAULT_MODEL_PATH
        if not target.exists():
            raise MLModelError(
                f"ML model file not found at {target}. "
                "Train it with: python scripts/train_ml_model.py"
            )
        payload = joblib.load(target)
        pipelines = payload.get("pipelines")
        metadata = payload.get("metadata") or {}
        if not isinstance(pipelines, dict) or not all(name in pipelines for name in CORE_VARIABLES):
            raise MLModelError("ML model file is missing one or more variable pipelines.")
        return cls(pipelines=pipelines, metadata=metadata)

    def predict_fields(self, report_text: str) -> list[FieldPrediction]:
        if not self.is_ready:
            raise MLModelError("ML extractor is not trained or loaded.")
        if not isinstance(report_text, str) or not report_text.strip():
            raise MLModelError("Report text must be a non-empty string.")

        predictions: list[FieldPrediction] = []
        for variable_name in CORE_VARIABLES:
            pipeline = self.pipelines[variable_name]
            proba = pipeline.predict_proba([report_text])[0]
            classes = list(pipeline.classes_)
            best_index = int(proba.argmax())
            label = str(classes[best_index])
            confidence = float(proba[best_index])
            probabilities = {
                str(class_name): float(score)
                for class_name, score in zip(classes, proba)
            }
            predictions.append(
                FieldPrediction(
                    variable_name=variable_name,
                    label=label,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            )
        return predictions


def _anchor_value(
    variable_name: str,
    value: str,
    report_text: str,
) -> tuple[str | None, EvidenceSpan | None, list[str]]:
    """Locate an exact evidence span for a predicted value without inventing text."""

    normalized = _normalize_value(variable_name, value)
    all_matches = _find_all_matches(report_text, PATTERNS[variable_name])
    active_matches, _ = _resolve_current_matches(report_text, all_matches)
    matching = [
        match
        for match in active_matches
        if _normalize_value(variable_name, match[1]).casefold() == normalized.casefold()
    ]
    if not matching:
        # Fall back to a direct lexical search for the predicted value.
        pattern = re.compile(re.escape(normalized), re.IGNORECASE)
        direct = [
            (match.group(0), match.group(0), match.start(), match.end())
            for match in pattern.finditer(report_text)
        ]
        matching = direct
    if not matching:
        return None, None, []
    best = _choose_best_match(matching, variable_name) or matching[0]
    evidence = _make_evidence(report_text, best[2], best[3])
    alternatives = sorted(
        {
            _normalize_value(variable_name, match[1])
            for match in active_matches
            if _normalize_value(variable_name, match[1]).casefold() != normalized.casefold()
        }
    )
    return normalized, evidence, alternatives


def _prediction_is_decisive(prediction: FieldPrediction) -> bool:
    """Return True when the top label is sufficiently preferred over alternatives."""

    if prediction.confidence < MIN_CONFIDENCE:
        return False
    ranked = sorted(prediction.probabilities.values(), reverse=True)
    if len(ranked) >= 2 and (ranked[0] - ranked[1]) < MIN_MARGIN:
        return False
    return True


def _manual_review_variable(
    variable_name: str,
    report_text: str,
    confidence_note: str,
    reason: str,
) -> tuple[VariableExtraction, list[QAIssue]]:
    status, status_evidence, note = _documented_no_value(variable_name, report_text)
    evidence = [status_evidence] if status_evidence else None
    description = f"{reason} {note}".strip()
    issue = QAIssue(
        issue_type=QAIssueType.MANUAL_REVIEW,
        variable_name=variable_name,  # type: ignore[arg-type]
        description=description,
        severity="medium",
        evidence=evidence,
        suggestion="Review the field manually; low-confidence or unanchored ML output was withheld.",
    )
    return (
        VariableExtraction(
            variable_name=variable_name,  # type: ignore[arg-type]
            extracted_value=None,
            documentation_status=DocumentationStatus.MANUAL_REVIEW_REQUIRED,
            evidence=evidence,
            notes=f"{confidence_note} {description}",
        ),
        [issue],
    )


def _variable_from_prediction(
    prediction: FieldPrediction,
    report_text: str,
) -> tuple[VariableExtraction, list[QAIssue]]:
    issues: list[QAIssue] = []
    variable_name = prediction.variable_name
    decoded_value, forced_status = _decode_label(prediction.label)
    confidence_note = f"ML confidence={prediction.confidence:.2f}."
    decisive = _prediction_is_decisive(prediction)

    if forced_status in {
        DocumentationStatus.NOT_DOCUMENTED,
        DocumentationStatus.CANNOT_BE_ASSIGNED,
        DocumentationStatus.CONFLICTING,
        DocumentationStatus.NEGATED,
    }:
        if not decisive:
            return _manual_review_variable(
                variable_name,
                report_text,
                confidence_note,
                f"ML special-state prediction was not decisive ({prediction.confidence:.2f}).",
            )

        if forced_status == DocumentationStatus.CONFLICTING:
            matches = _find_all_matches(report_text, PATTERNS[variable_name])
            active, _ = _resolve_current_matches(report_text, matches)
            paired: list[tuple[str, EvidenceSpan]] = []
            seen: set[str] = set()
            for match in active:
                value = _normalize_value(variable_name, match[1])
                key = value.casefold()
                if key in seen:
                    continue
                span = _make_evidence(report_text, match[2], match[3])
                if span is None:
                    continue
                seen.add(key)
                paired.append((value, span))
                if len(paired) >= 4:
                    break
            if len(paired) >= 2:
                alternatives = [value for value, _ in paired]
                evidence = [span for _, span in paired]
                note = "ML predicted unresolved conflicting documentation."
                issues.append(
                    QAIssue(
                        issue_type=QAIssueType.CONFLICTING,
                        variable_name=variable_name,  # type: ignore[arg-type]
                        description=note,
                        severity="high",
                        evidence=evidence,
                        suggestion="Resolve the alternatives manually; no ML value was selected.",
                    )
                )
                return (
                    VariableExtraction(
                        variable_name=variable_name,  # type: ignore[arg-type]
                        extracted_value=None,
                        documentation_status=DocumentationStatus.CONFLICTING,
                        evidence=evidence,
                        alternatives=alternatives,
                        notes=f"{confidence_note} {note}",
                    ),
                    issues,
                )
            return _manual_review_variable(
                variable_name,
                report_text,
                confidence_note,
                "ML predicted conflict but could not pair distinct evidence-backed alternatives.",
            )

        status, status_evidence, note = _documented_no_value(variable_name, report_text)
        # Prefer the ML-forced semantic status when the report supports a no-value state.
        final_status = (
            forced_status
            if status_evidence or forced_status != DocumentationStatus.NOT_DOCUMENTED
            else status
        )
        evidence = [status_evidence] if status_evidence else None
        if final_status == DocumentationStatus.NEGATED and not evidence:
            matches = _find_all_matches(report_text, PATTERNS[variable_name])
            if matches:
                _, value, start, end = matches[0]
                evidence_span = _negation_evidence(variable_name, value, report_text, start, end)
                evidence = [evidence_span] if evidence_span else None
        if final_status == DocumentationStatus.NEGATED and not evidence:
            return _manual_review_variable(
                variable_name,
                report_text,
                confidence_note,
                "ML predicted negation but no exact negation evidence was found.",
            )
        if final_status == DocumentationStatus.CANNOT_BE_ASSIGNED and not evidence:
            return _manual_review_variable(
                variable_name,
                report_text,
                confidence_note,
                "ML predicted cannot-be-assigned without exact supporting evidence.",
            )
        issue_type = (
            QAIssueType.NEGATED
            if final_status == DocumentationStatus.NEGATED
            else QAIssueType.MISSING
        )
        issues.append(
            QAIssue(
                issue_type=issue_type,
                variable_name=variable_name,  # type: ignore[arg-type]
                description=note,
                severity=(
                    "high"
                    if variable_name in {"histologic_diagnosis", "pathological_n_category"}
                    else "medium"
                ),
                evidence=evidence,
                suggestion="Confirm the documentation manually; do not infer a category.",
            )
        )
        return (
            VariableExtraction(
                variable_name=variable_name,  # type: ignore[arg-type]
                extracted_value=None,
                documentation_status=final_status,
                evidence=evidence,
                notes=f"{confidence_note} {note}",
            ),
            issues,
        )

    assert decoded_value is not None
    value, evidence, _unused_alternatives = _anchor_value(
        variable_name, decoded_value, report_text
    )
    if value is None or evidence is None:
        # Safety gate: never invent values. If the model is indecisive, prefer review.
        if not decisive:
            return _manual_review_variable(
                variable_name,
                report_text,
                confidence_note,
                f"ML prediction was not decisive ({prediction.confidence:.2f}) and no exact anchor was found.",
            )
        note = (
            f"ML predicted '{decoded_value}' but could not anchor an exact evidence span "
            "in the report, so the value was withheld."
        )
        issues.append(
            QAIssue(
                issue_type=QAIssueType.UNSUPPORTED_EVIDENCE,
                variable_name=variable_name,  # type: ignore[arg-type]
                description=note,
                severity="critical",
                suggestion="Do not accept unanchored ML values.",
            )
        )
        return (
            VariableExtraction(
                variable_name=variable_name,  # type: ignore[arg-type]
                extracted_value=None,
                documentation_status=DocumentationStatus.UNSUPPORTED,
                evidence=None,
                notes=f"{confidence_note} {note}",
            ),
            issues,
        )

    # Exact anchoring is the primary safety gate. Low absolute multiclass
    # probabilities are expected on the tiny synthetic training set.
    negation = _negation_evidence(
        variable_name, value, report_text, evidence.start_offset, evidence.end_offset
    )
    if negation is not None:
        note = "ML candidate is explicitly negated in the report."
        issues.append(
            QAIssue(
                issue_type=QAIssueType.NEGATED,
                variable_name=variable_name,  # type: ignore[arg-type]
                description=note,
                severity="high" if variable_name == "histologic_diagnosis" else "medium",
                evidence=[negation],
                suggestion="Confirm the negated finding during review.",
            )
        )
        return (
            VariableExtraction(
                variable_name=variable_name,  # type: ignore[arg-type]
                extracted_value=None,
                documentation_status=DocumentationStatus.NEGATED,
                evidence=[negation],
                notes=f"{confidence_note} {note}",
            ),
            issues,
        )

    if value in {"pNX", "pTX"}:
        status = DocumentationStatus.CANNOT_BE_ASSIGNED
        note = (
            "Explicit pNX documented; it is distinct from pN0 and from an undocumented pN."
            if value == "pNX"
            else "Explicit pTX documented; no pathological T category was inferred."
        )
        issues.append(
            QAIssue(
                issue_type=QAIssueType.MANUAL_REVIEW,
                variable_name=variable_name,  # type: ignore[arg-type]
                description=note,
                severity="medium",
                evidence=[evidence],
                suggestion=(
                    "Retain pNX as explicitly documented; do not convert it to pN0."
                    if value == "pNX"
                    else "Retain pTX as explicitly documented; do not infer pT from size."
                ),
            )
        )
        return (
            VariableExtraction(
                variable_name=variable_name,  # type: ignore[arg-type]
                extracted_value=value,
                documentation_status=status,
                evidence=[evidence],
                notes=f"{confidence_note} {note}",
            ),
            issues,
        )

    context = _local_context(report_text, evidence.start_offset, evidence.end_offset)
    uncertain = forced_status == DocumentationStatus.UNCERTAIN or _has_uncertainty(context)
    if uncertain:
        note = "ML value is qualified by uncertainty language or an uncertain gold label."
        issues.append(
            QAIssue(
                issue_type=QAIssueType.UNCERTAIN,
                variable_name=variable_name,  # type: ignore[arg-type]
                description=note,
                severity="medium",
                evidence=[evidence],
                suggestion="Review the candidate before acceptance.",
            )
        )
        return (
            VariableExtraction(
                variable_name=variable_name,  # type: ignore[arg-type]
                extracted_value=value,
                documentation_status=DocumentationStatus.UNCERTAIN,
                evidence=[evidence],
                notes=f"{confidence_note} {note}",
            ),
            issues,
        )

    note = "ML candidate is anchored to exact report evidence."
    return (
        VariableExtraction(
            variable_name=variable_name,  # type: ignore[arg-type]
            extracted_value=value,
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[evidence],
            notes=f"{confidence_note} {note}",
        ),
        issues,
    )


def ml_extract(
    report_text: str,
    report_id: str,
    model: PathologyMLExtractor | None = None,
) -> ExtractionResult:
    """Run the local ML extractor and return a schema-compatible result."""

    extractor = model or get_default_ml_extractor()
    predictions = extractor.predict_fields(report_text)
    variables: list[VariableExtraction] = []
    issues: list[QAIssue] = []
    for prediction in predictions:
        variable, variable_issues = _variable_from_prediction(prediction, report_text)
        variables.append(variable)
        issues.extend(variable_issues)
    priority, reason = _issue_priority(issues)
    return ExtractionResult(
        report_id=report_id,
        method="ml",
        variables=variables,
        qa_issues=issues,
        manual_review_required=priority != "low",
        review_priority=priority,
        review_priority_reason=reason,
        timestamp=get_iso_timestamp(),
    )


@lru_cache(maxsize=1)
def get_default_ml_extractor() -> PathologyMLExtractor:
    """Load the on-disk model, or train and persist one from bundled synthetic data."""

    if DEFAULT_MODEL_PATH.exists():
        return PathologyMLExtractor.load(DEFAULT_MODEL_PATH)
    extractor = PathologyMLExtractor().train()
    extractor.save(DEFAULT_MODEL_PATH)
    get_default_ml_extractor.cache_clear()
    return PathologyMLExtractor.load(DEFAULT_MODEL_PATH)


def reset_ml_extractor_cache() -> None:
    get_default_ml_extractor.cache_clear()
