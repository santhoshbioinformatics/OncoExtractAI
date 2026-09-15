# Demonstration data boundary

Only these files are application fixtures and safe to include with the prototype:

- `synthetic_reports.json` — eight visibly watermarked synthetic reports
- `sample_gold_annotations.csv` — synthetic reference annotations
- `annotation_template.csv` — blank annotation template

`raw/` and `processed/` are ignored local-data locations. They are not read by
the application, evaluation, tests, or export workflow and must be excluded
from archives, commits, screenshots, and shared demonstrations.
