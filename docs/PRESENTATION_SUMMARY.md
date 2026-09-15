# OncoExtractAI-QA — Presentation Summary

## Slide 1 — Problem and scope

- Lung pathology reports express key abstraction variables in free text.
- Missing, uncertain, conflicting, negated, and amended statements make direct extraction unsafe.
- The prototype handles four fields: histologic diagnosis, tumor size, pT, and pN.
- It is a local research and review assistant, not a diagnostic or clinical staging product.

## Slide 2 — Evidence-first workflow

```text
Approved text
  → baseline or evidence-first extraction
  → schema and exact-evidence validation
  → documentation QA and review priority
  → human decision
  → audit event
  → reviewed JSON/CSV export
```

- Evidence-first proposed values must point to exact report substrings.
- Offsets are zero-based and end-exclusive.
- Missing or unresolved evidence leads to abstention or manual review.

## Slide 3 — Clinical guardrails

- Never infer pT or pN from measurements, node counts, or staging rules; return a category only when the report explicitly states it.
- Keep explicit `pN0`, explicit `pNX`, and `not_documented` distinct.
- Retain `pNX` as the value `pNX` with status `cannot_be_assigned`.
- Prefer an amended/final value only when the text explicitly establishes that it supersedes the earlier value.
- Treat conflict and uncertainty as documentation states, not invitations to guess.

## Slide 4 — Eight synthetic scenarios

- Complete supported abstraction (`SYN-LUAD-001`)
- No nodes submitted / pN cannot be assigned (`SYN-LUSC-002`)
- pN truly not documented (`SYN-LUAD-003`)
- Explicit pNX (`SYN-LUSC-004`)
- Explicitly superseded preliminary size and pT (`SYN-LUAD-005`)
- Unresolved size/pT/pN conflicts (`SYN-LUSC-006`)
- Uncertain diagnosis/size and provisional pT (`SYN-LUAD-007`)
- Explicitly negated malignancy (`SYN-LUSC-008`)

Every report is visibly watermarked, carries `is_synthetic: true`, and omits patient names and medical-record numbers.

## Slide 5 — Human review, audit, and export

- Reviewers accept, correct, reject, or flag each field.
- Correction, rejection, and flag actions require a reason.
- Audit events preserve the original output, action, corrected value, timestamp, and review reason.
- Reviewed JSON and CSV exports include the final abstraction and provenance needed for follow-up.
- Spreadsheet-safe serialization prevents reviewer-entered text from becoming a formula when CSV is opened.

## Slide 6 — Verification, not a performance claim

- The acceptance set contains eight reports and 32 annotated fields.
- Automated validation checks report/annotation uniqueness, status values, pN distinctions, CSV structure, and every populated evidence offset.
- Tests cover input handling, extraction schemas, exact evidence, QA states, review actions, audit records, and export behavior.
- Any dashboard or export must be labeled **Illustrative synthetic results** and show its denominator.
- Handcrafted synthetic results do not establish clinical accuracy, safety, or efficiency.

## Slide 7 — Limitations and next milestone

- Rule/regex extraction remains sensitive to wording and section layout.
- The small synthetic set has no independent annotator or adjudication.
- The prototype has no production authentication, clinical governance, or validated deployment boundary.
- Next: a blinded, prespecified evaluation on a larger approved dataset with two-reviewer adjudication, exact evidence-offset scoring, and measured review workload.

## 30-second summary

OncoExtractAI-QA is a synthetic-data research prototype for transparent lung pathology abstraction. It compares baseline and evidence-first extraction, requires exact source evidence for proposed values, preserves clinically important documentation states, abstains when the report cannot support an answer, and keeps human decisions traceable through audit-ready reviewed exports. Its current outputs are illustrative workflow checks—not clinical performance estimates.
