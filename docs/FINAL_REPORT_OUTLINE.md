# OncoExtractAI-QA — Final Report Outline

**Project:** Evidence-grounded, human-in-the-loop abstraction from lung pathology text  
**Prototype:** Local Streamlit research application using eight synthetic LUAD/LUSC reports  
**Safety statement:** Not for diagnosis, treatment, staging decisions, or clinical use

## 1. Abstract

- Problem: pathology variables are difficult to abstract reliably from free text, especially when documentation is missing, conflicting, uncertain, negated, or amended.
- Approach: compare a rule-based baseline with an evidence-first QA workflow for histologic diagnosis, tumor size, pT, and pN.
- Safety design: require exact evidence for proposed values, preserve distinct documentation states, abstain when appropriate, and route difficult cases to human review.
- Setting: eight visibly watermarked synthetic reports and 32 reference annotations.
- Contribution: a transparent local workflow joining extraction, evidence inspection, review decisions, audit history, and export.

## 2. Motivation and research questions

1. Can the workflow return each proposed value with an exact, inspectable evidence span?
2. Can it distinguish supported, missing, cannot-be-assigned, negated, uncertain, conflicting, and superseded documentation?
3. Does it preserve the clinically important distinction among explicitly documented `pN0`, explicit `pNX`, and `not_documented`?
4. Can a reviewer correct or reject output while retaining a useful audit record and reviewed export?

## 3. Scope and safety boundaries

- In scope: synthetic LUAD/LUSC pathology text and four abstraction variables.
- Out of scope: diagnosis, treatment advice, prognosis, radiology, NGS, EHR integration, automated staging, and patient-facing use.
- No real-world clinical dataset is an active application feature.
- The prototype is not a HIPAA-compliant storage or access-control system.

### Non-inference rule

The system may return pT or pN only when the report explicitly states that category. It must not derive a category from tumor size, node counts, anatomic findings, or staging guidelines. Clinically plausible synthetic labels make the fixtures credible; they do not authorize software-based staging inference.

## 4. Synthetic acceptance set

Every report starts with `SYNTHETIC DEMONSTRATION REPORT — NO REAL PATIENT DATA`, uses a `SYN-` identifier, has `is_synthetic: true`, and omits patient names and medical-record numbers.

| Report | Primary scenario |
| --- | --- |
| `SYN-LUAD-001` | Complete supported abstraction |
| `SYN-LUSC-002` | No nodes submitted; pN cannot be assigned |
| `SYN-LUAD-003` | pN truly not documented |
| `SYN-LUSC-004` | Explicit pNX retained as `pNX` |
| `SYN-LUAD-005` | Addendum explicitly supersedes preliminary size and pT |
| `SYN-LUSC-006` | Unresolved conflicting size, pT, and pN |
| `SYN-LUAD-007` | Uncertain diagnosis/size and provisional pT |
| `SYN-LUSC-008` | Explicitly negated malignancy |

These handcrafted reports are a functional acceptance set, not a representative patient cohort or a clinical validation dataset.

## 5. System workflow

1. Select a synthetic report or enter text approved for local research use.
2. Run baseline or evidence-first extraction.
3. Validate the structured result.
4. Check each populated evidence span against the exact source substring and zero-based, end-exclusive offsets.
5. Detect documentation issues and assign an understandable review priority.
6. Let the reviewer accept, correct, reject, or flag every field.
7. Append an audit event containing original output, reviewer action, corrected value, timestamp, and reason.
8. Export the reviewed abstraction and audit information as JSON or CSV.

## 6. Documentation semantics

- `supported`: an explicit value has valid evidence.
- `not_documented`: no value is present; the system abstains without inventing evidence.
- `cannot_be_assigned`: the report explicitly says the category cannot be assessed or assigned; explicit `pNX` remains the value `pNX`.
- `negated`: the candidate finding is explicitly denied.
- `uncertain`: the report qualifies the proposed value as favored, estimated, provisional, or otherwise uncertain.
- `conflicting`: incompatible values are present and no final value is designated.
- `superseded`: an earlier value is explicitly replaced by a current amended/revised value.
- `unsupported` and `manual_review_required`: safety outcomes used when evidence or automated resolution is insufficient.

For the amended fixture, the revised 4.1 cm and pT2b values are the supported current reference values. The earlier 2.8 cm and pT1c entries are represented as superseded issues, not as the gold current values.

## 7. Verification and evaluation plan

### Fixture integrity

Run:

```bash
python scripts/validate_demo_data.py
python -m pytest -q
```

The validator checks:

- exactly eight reports and all eight required scenarios;
- visible synthetic labels and absence of direct-identifier labels;
- exactly four unique annotations per report;
- valid CSV shape, statuses, and review booleans;
- pN0/pNX/not-documented/cannot-assign distinctions;
- exact character offsets for every populated evidence span;
- a well-formed annotation template.

### Prototype checks

- Exact value match by variable.
- Documentation-status accuracy.
- Exact evidence-text and offset validity.
- Appropriate abstention for absent or unresolved values.
- Review referral counts with reasons and transparent denominators.
- Reviewer correction, rejection, audit, and JSON/CSV export behavior.

Any dashboard or exported summary must be labeled **Illustrative synthetic results** and state the denominator: eight reports and 32 fields. Results must not be described as clinical accuracy or generalized beyond these fixtures. Report only checks and commands that were actually run.

## 8. Implementation

- UI and session workflow: `app.py`
- Types and validation: `src/schemas.py`
- Input normalization and direct-identifier checks: `src/report_input.py`
- Extraction/validation/QA orchestration: `src/workflow.py`
- Extraction: `src/extractor.py`
- Evidence checks: `src/evidence_validator.py`
- Documentation QA and priority: `src/qa_detector.py`
- Evaluation: `src/evaluation.py`
- Safe display and highlighting: `src/display.py`
- Review state and append-only session audit events: `src/review_workflow.py`
- Reviewed JSON/CSV serialization: `src/exporter.py`
- Synthetic-fixture and gold-annotation loading: `src/utils.py`
- Synthetic fixtures and reference labels: `data/`
- Automated checks: `tests/` and `scripts/validate_demo_data.py`

## 9. Limitations

- Eight handcrafted reports cannot establish real-world validity, safety, or efficiency.
- The reference labels have no independent second annotator or adjudication.
- Rule/regex extraction remains sensitive to wording and section layout.
- The local prototype lacks production authentication, access control, retention policy, and clinical governance.

## 10. Recommended next milestone

Conduct a blinded study on a larger, approved dataset with two-reviewer annotation and adjudication. Prespecify field-level value, status, evidence-offset, abstention, and review-workload measures; retain the same non-inference and evidence gates; and measure reviewer time without weakening privacy controls.

## 11. Appendix artifacts

- `data/synthetic_reports.json`
- `data/sample_gold_annotations.csv`
- `data/annotation_template.csv`
- `scripts/validate_demo_data.py`
- Unit-test output and command log
- Desktop/mobile visual-check record; screenshots only when the browser surface was available and actually inspected
- Example reviewed JSON and CSV exports with no sensitive text
