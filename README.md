# OncoExtractAI-QA

OncoExtractAI-QA is a local, evidence-grounded review assistant for abstracting four variables from lung pathology text: histologic diagnosis, tumor size, pathological T category, and pathological N category. It compares a conventional rule-based baseline, an evidence-first QA workflow, and a local machine-learning extractor, shows supporting text, surfaces documentation problems, supports human review, and exports reviewed results.

> **Research prototype — not for diagnosis, treatment, staging decisions, or clinical use.** The bundled demonstration reports are wholly synthetic and contain no real patient data. Evaluation results derived from them must be labeled **Illustrative synthetic results**.

## Quick start

Prerequisites: Python 3.10 or newer and local Tesseract OCR. On macOS install
Tesseract with `brew install tesseract`; on Ubuntu use
`sudo apt install tesseract-ocr`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Authentication is intentionally disabled for the current local research prototype.
Run it only on a trusted machine or protected network; do not expose the Streamlit
server directly to the public internet. Add organization-managed authentication
before deployment or use with approved non-synthetic text.

## Review workflow

Completed reviews retain field-level provenance: extraction method and a hash-based
implementation version, the SHA-256 source report digest, extraction/review timestamps,
session-entered reviewer identity, original and corrected values, exact evidence offsets,
and the review reason. Authentication is currently disabled, so reviewer identities are
explicitly labeled unverified in the interface and exports.

1. Add a bundled synthetic report, approved text, or an approved PDF (25 MB and 100 pages maximum).
2. For PDFs, verify every page in the split OCR & text review. Native PDF text is preferred;
   poor pages fall back to local Tesseract OCR. Abstraction remains locked until every page is accepted.
3. Work through the filterable review queue by status, priority, method, and session-only assignment.
4. Run baseline, evidence-first, or ML extraction on the accepted authoritative text.
5. Review exception fields first in the split-screen report and decision workspace.
6. Accept, correct, reject, or flag each field; drafts autosave within the current session.
7. Complete the review, then export the full session or one reviewed case as JSON or CSV.

PDF bytes, rendered page images, extracted text, and reviewer corrections are kept only
in Streamlit session memory. The app does not infer metadata from filenames and does not
send OCR content to a remote service. Clear or discard a document when finished, close the
browser session on a shared workstation, and do not commit document artifacts.

The empty workspace offers a one-click synthetic example and describes the next useful
action at each stage. In Human Review, keyboard shortcuts support `A` accept, `C`
correct, `F` flag, `J`/`K` next or previous field, `E` evidence focus, and
`Cmd/Ctrl + Enter` complete review. Letter shortcuts are suppressed while typing.

Synthetic evaluation is kept under the separate **Research tools** sidebar section so
it does not compete with daily review. Because authentication is disabled, that section
is only organizational in the local prototype; enforce authenticated role-based access
before deployment.

The evidence-first path is designed to abstain or request manual review when the report cannot support a reliable value. A human reviewer remains responsible for every accepted abstraction.

## Clinical and evidence guardrails

- `pN0`, `pNX`, and `not_documented` are distinct states. `pN0` requires an explicitly documented category; `pNX` is retained as the explicit value while its status is `cannot_be_assigned`; `not_documented` means no pN value is present.
- The extractor must never calculate or infer pT or pN from tumor measurements, node counts, or other findings. It may return a category only when that category is explicitly stated in the report.
- Evidence must be an exact substring of the submitted report. Character offsets use a zero-based, end-exclusive interval: `report_text[start_offset:end_offset]`.
- A revised or amended result takes precedence only when the report explicitly says that it supersedes the earlier result.
- Missing, contradictory, provisional, or insufficient evidence leads to abstention or manual review rather than a fabricated conclusion.
- Confidence values, evidence, clinical conclusions, and evaluation results must never be invented or presented as calibrated unless their method is documented and validated.

Supported documentation statuses are `supported`, `not_documented`, `cannot_be_assigned`, `negated`, `uncertain`, `conflicting`, `superseded`, `unsupported`, and `manual_review_required`.

## Synthetic acceptance set

[`data/synthetic_reports.json`](data/synthetic_reports.json) contains eight visibly watermarked reports. Every report has `is_synthetic: true`, a `SYN-` identifier, no patient name or medical-record number, and one primary QA scenario.

| Report | Scenario |
| --- | --- |
| `SYN-LUAD-001` | Complete, supported abstraction |
| `SYN-LUSC-002` | No nodes submitted; pN cannot be assigned |
| `SYN-LUAD-003` | pN truly not documented |
| `SYN-LUSC-004` | Explicit pNX retained as `pNX` |
| `SYN-LUAD-005` | Explicit addendum superseding preliminary size and pT |
| `SYN-LUSC-006` | Unresolved conflicting size, pT, and pN |
| `SYN-LUAD-007` | Uncertain diagnosis/size and provisional pT |
| `SYN-LUSC-008` | Explicitly negated malignancy |

The reports are a small functional acceptance set, not a representative clinical cohort and not evidence of real-world performance. Their pT labels are synthetic statements reviewed for internal clinical plausibility; the software copies explicitly stated categories and does not derive them from size. The fixture terminology follows the [CAP lung resection protocol](https://documents.cap.org/protocols/cp-thorax-lung-resection-20-4101.pdf) and [IASLC eighth-edition staging summary](https://www.iaslc.org/research-education/publications-resources-guidelines/summary-8th-edition-tnm-classification).

Reference annotations live in [`data/sample_gold_annotations.csv`](data/sample_gold_annotations.csv). They contain exactly four unique rows per report. Every populated evidence span has exact offsets; `not_documented` rows intentionally abstain without an evidence span. Validate the JSON, CSV shape, scenario coverage, synthetic watermark, pN distinctions, and every populated evidence slice with:

```bash
python scripts/validate_demo_data.py
```

## Evaluation

The dashboard compares the two extraction methods against the synthetic reference annotations. Results from eight handcrafted reports (32 fields) are useful for regression testing and workflow demonstration only. They are not estimates of clinical accuracy, should not be generalized beyond these fixtures, and should always display the label **Illustrative synthetic results** with their denominators.

The same page reports session-based reviewer-efficiency measures after reviews are
completed: median review time per report, field acceptance and correction rates,
reviewer disagreement actions, clarification returns, and reviews completed without
editing. Timing begins with the first changed reviewer decision and ends at completion.
These workflow measures are descriptive and are not clinical-performance claims.

The primary research metric is the unsupported extraction rate:

```text
non-abstained returned values without valid exact evidence
---------------------------------------------------------
              all non-abstained returned values
```

Secondary checks should report exact value match, documentation-status accuracy, exact evidence-and-offset validity, appropriate abstention, coverage, and manual-review referral with transparent numerators and denominators. Do not interpret schema validity or referral volume alone as clinical correctness.

Rates with a zero denominator are displayed and exported as `N/A`, never as a fabricated 0%. Referral precision and sensitivity use the report-level gold review label (`any` field marked for review) and are reported separately from raw referral volume.

## Architecture

```text
app.py                         Streamlit workspace and review flow
src/schemas.py                 Validated extraction, evidence, QA, and review models
src/report_input.py            Input normalization, validation, and direct-identifier checks
src/pdf_processor.py           PDF validation, native extraction, page rendering, and routing
src/ocr_service.py             Local-only Tesseract OCR adapter
src/text_normalizer.py         Conservative text cleanup and transformation provenance
src/workflow.py                Extraction, evidence validation, and QA orchestration
src/extractor.py               Baseline and evidence-first extraction
src/ml_model.py                Local TF-IDF + logistic regression extractor
src/evidence_validator.py      Exact substring and character-offset checks
src/qa_detector.py             Documentation issue detection and review priority
src/evaluation.py              Synthetic reference-set evaluation
src/display.py                 Safe report rendering and evidence highlighting
src/review_workflow.py         Review-state validation and append-only session audit events
src/exporter.py                Reviewed JSON/CSV serialization and spreadsheet safety
src/utils.py                   Synthetic fixture and gold-annotation loading
data/synthetic_reports.json    Synthetic demonstration reports
data/sample_gold_annotations.csv
models/                        Local trained joblib artifacts (retrainable)
scripts/validate_demo_data.py  Fixture and annotation integrity check
scripts/train_ml_model.py      Train/save the local ML extractor
tests/                         Unit tests
outputs/                       Local generated artifacts; ignored by version control
```

UI, extraction logic, evidence validation, QA detection, review state, evaluation, and export are separated by module. Structured extraction output is validated before display; deterministic evidence validation is an additional gate rather than a substitute for human review.

### Local ML extractor

The ML path in [`src/ml_model.py`](src/ml_model.py) is a real offline model, not a remote LLM:

1. Character n-gram TF-IDF + logistic regression predicts each core variable label.
2. Predicted values are accepted only when an exact report substring can be anchored.
3. Evidence validation and QA detection still run after ML extraction.
4. Clinical guardrails still refuse invented pT/pN categories and unanchored values.

Train or refresh the model from the bundled synthetic gold set:

```bash
python scripts/train_ml_model.py
```

The first ML extraction also auto-trains and saves `models/pathology_ml_extractor.joblib` if the file is missing. This model is a research demonstration trained on eight synthetic reports; it is not a clinical system.

## Verification

Run the checks supported by this repository from its root:

```bash
python scripts/validate_demo_data.py
python -m pytest -q
python -m compileall -q app.py src scripts tests
```

Then start Streamlit and manually verify report entry, both extraction methods, evidence navigation, documentation issues, review actions, audit history, and JSON/CSV downloads at desktop and mobile widths. Do not report a command or visual check as passing unless it was actually run.

## Privacy and scope

- Use only the bundled synthetic fixtures or reports explicitly approved for the local research workflow.
- Do not commit or redistribute raw clinical text, direct identifiers, TCGA participant barcodes, local exports, or review notes.
- This prototype is not a HIPAA-compliant storage, access-control, or audit system. Review browser downloads and local output files before sharing them.
- NGS, radiology, prognosis, treatment recommendations, EHR integration, patient-facing use, and automated clinical staging are out of scope.

Local raw/processed data paths and generated outputs are ignored by [`.gitignore`](.gitignore). No real-world dataset is bundled as an active application feature.
The legacy `scripts/prepare_tcga_lung_reports.py` entry point now fails closed and cannot download, label, or load patient-derived reports. Only the three files named in [`data/README.md`](data/README.md) belong in a shared prototype package.

## Limitations and next milestone

- Rule/regex extraction is intentionally narrow and remains sensitive to wording and section structure.
- The eight-report synthetic set has no independent annotator, adjudication, inter-rater reliability, or external validation.
- Review state and exports are prototype-grade and require further security and provenance work before use with approved non-synthetic text.
- No clinical performance, efficiency benefit, or safety claim has been established.

The recommended next milestone is a blinded evaluation on a larger, independently annotated, approved dataset with two-reviewer adjudication, strict provenance, prespecified metrics, and measured reviewer time—while preserving the same evidence and abstention gates.
