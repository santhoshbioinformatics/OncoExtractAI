#!/usr/bin/env python3
"""Legacy entry point retained as a fail-closed safety notice.

OncoExtractAI-QA is a synthetic-only demonstration. The former implementation
downloaded and heuristically relabeled a patient-derived TCGA report corpus.
That workflow is intentionally disabled: it was outside the prototype's data
boundary and its barcode-prefix mapping was not an authoritative cohort label.
"""

from __future__ import annotations

import sys


NOTICE = (
    "Disabled: this research prototype only supports the visibly synthetic "
    "fixtures in data/synthetic_reports.json. No patient-derived dataset is "
    "downloaded, prepared, or loaded."
)


def main() -> int:
    print(NOTICE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
