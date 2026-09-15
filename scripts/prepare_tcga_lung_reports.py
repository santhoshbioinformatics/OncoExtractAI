#!/usr/bin/env python3
"""Download and prepare TCGA lung pathology reports for OncoExtractAI-QA.

This script downloads pathology reports from the GDC API for TCGA-LUAD and
TCGA-LUSC cohorts, extracts text content, and outputs them in the project's
canonical format for local research use.

Usage:
    python scripts/prepare_tcga_lung_reports.py
    python scripts/prepare_tcga_lung_reports.py --max-reports 50
    python scripts/prepare_tcga_lung_reports.py --output data/raw/tcga_reports.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

# Project root
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# GDC API endpoints
GDC_API = "https://api.gdc.cancer.gov"
CASES_ENDPOINT = f"{GDC_API}/cases"
FILES_ENDPOINT = f"{GDC_API}/files"
DATA_ENDPOINT = f"{GDC_API}/data"

# TCGA lung cancer project IDs
TCGA_LUNG_PROJECTS = ["TCGA-LUAD", "TCGA-LUSC"]

# Cancer type mapping
PROJECT_TO_CANCER_TYPE = {
    "TCGA-LUAD": "LUAD",
    "TCGA-LUSC": "LUSC",
}

# Heuristic patterns for labeling
DIAGNOSIS_PATTERNS = {
    "LUAD": [
        r"(?i)invasive\s+adenocarcinoma",
        r"(?i)adenocarcinoma",
        r"(?i)acinar\s+adenocarcinoma",
        r"(?i)papillary\s+adenocarcinoma",
        r"(?i)micropapillary\s+adenocarcinoma",
        r"(?i)solid\s+adenocarcinoma",
        r"(?i)lepidic\s+adenocarcinoma",
    ],
    "LUSC": [
        r"(?i)invasive\s+squamous\s+cell\s+carcinoma",
        r"(?i)squamous\s+cell\s+carcinoma",
        r"(?i)non-keratinizing\s+squamous",
        r"(?i)keratinizing\s+squamous",
    ],
}

SIZE_PATTERN = r"(?i)(?:tumor\s+size|greatest\s+dimension|gross\s+size|measuring|measures?)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)"
PT_PATTERN = r"(?i)\b(pT(?:0|is|1mi|1[abc]?|2[ab]?|3|4))\b"
PN_PATTERN = r"(?i)\b(pN(?:X|[0-3]))\b"


class TCGAReport:
    """Container for a TCGA pathology report."""

    def __init__(
        self,
        case_id: str,
        project_id: str,
        cancer_type: str,
        report_id: str,
        text: str,
        source_file: str,
        download_date: str,
    ):
        self.case_id = case_id
        self.project_id = project_id
        self.cancer_type = cancer_type
        self.report_id = report_id
        self.text = text
        self.source_file = source_file
        self.download_date = download_date

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "case_id": self.case_id,
            "project_id": self.project_id,
            "cancer_type": self.cancer_type,
            "text": self.text,
            "source_file": self.source_file,
            "download_date": self.download_date,
            "is_synthetic": False,
            "source": "tcga",
        }


def query_gdc_cases(
    project_ids: list[str],
    max_cases: int = 0,
) -> list[dict[str, Any]]:
    """Query GDC API for lung cancer cases."""
    print(f"Querying GDC API for cases in projects: {', '.join(project_ids)}")

    cases = []
    filters = {
        "op": "in",
        "content": [
            {"field": "project.project_id", "value": pid}
            for pid in project_ids
        ],
    }

    case_filters = {
        "op": "and",
        "content": [
            filters,
            {
                "op": ">",
                "content": {"field": "diagnoses.days_to_birth"},
                "value": 0,
            },
        ],
    }

    params = {
        "filters": json.dumps(case_filters),
        "format": "json",
        "size": min(max_cases or 200, 200),
    }

    try:
        response = requests.post(CASES_ENDPOINT, json=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        cases = data.get("data", {}).get("hits", [])
        print(f"Found {len(cases)} cases")
    except requests.RequestException as e:
        print(f"Error querying GDC API: {e}", file=sys.stderr)
        sys.exit(1)

    return cases


def get_pathology_files(case_id: str) -> list[dict[str, Any]]:
    """Get pathology report files for a case."""
    filters = {
        "op": "and",
        "content": [
            {"field": "cases.case_id", "value": case_id},
            {"field": "file_format", "value": "pdf"},
        ],
    }

    text_filters = {
        "op": "and",
        "content": [
            {"field": "cases.case_id", "value": case_id},
            {
                "op": "in",
                "content": [
                    {"field": "file_format", "value": "txt"},
                    {"field": "file_format", "value": "text"},
                ],
            },
        ],
    }

    files = []
    for f in [filters, text_filters]:
        try:
            response = requests.post(
                FILES_ENDPOINT,
                json={"filters": json.dumps(f), "format": "json", "size": 100},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            files.extend(data.get("data", {}).get("hits", []))
        except requests.RequestException:
            pass

    return files


def download_file(file_id: str) -> bytes | None:
    """Download a file from GDC."""
    try:
        response = requests.get(
            f"{DATA_ENDPOINT}/{file_id}",
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"  Warning: Could not download file {file_id}: {e}", file=sys.stderr)
        return None


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        print("  Warning: pypdf not installed. Install with: pip install pypdf", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  Warning: Could not extract text from PDF: {e}", file=sys.stderr)
        return ""


def extract_tumor_size(text: str) -> str | None:
    """Extract tumor size from report text."""
    match = re.search(SIZE_PATTERN, text)
    if match:
        size = float(match.group(1))
        return f"{size:.1f} cm"
    return None


def extract_pt_category(text: str) -> str | None:
    """Extract pT category from report text."""
    match = re.search(PT_PATTERN, text)
    if match:
        return match.group(1)
    return None


def extract_pn_category(text: str) -> str | None:
    """Extract pN category from report text."""
    match = re.search(PN_PATTERN, text)
    if match:
        return match.group(1)
    return None


def generate_gold_annotations(reports: list[TCGAReport]) -> list[dict[str, Any]]:
    """Generate heuristic gold annotations for TCGA reports."""
    annotations = []

    for report in reports:
        text = report.text
        report_id = report.report_id
        cancer_type = report.cancer_type

        # Histologic diagnosis
        diagnosis = None
        for pattern in DIAGNOSIS_PATTERNS.get(cancer_type, []):
            match = re.search(pattern, text)
            if match:
                diagnosis = match.group(0).strip()
                break

        if diagnosis:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "histologic_diagnosis",
                "gold_value": diagnosis,
                "evidence_text": diagnosis,
                "evidence_start_offset": str(text.find(diagnosis)),
                "evidence_end_offset": str(text.find(diagnosis) + len(diagnosis)),
                "documentation_status": "supported",
                "manual_review_required": "True",
                "notes": "Heuristic extraction from TCGA report - requires manual review",
            })
        else:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "histologic_diagnosis",
                "gold_value": "",
                "evidence_text": "",
                "evidence_start_offset": "",
                "evidence_end_offset": "",
                "documentation_status": "not_documented",
                "manual_review_required": "True",
                "notes": "Diagnosis not found by heuristic - requires manual review",
            })

        # Tumor size
        size = extract_tumor_size(text)
        if size:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "tumor_size",
                "gold_value": size,
                "evidence_text": size,
                "evidence_start_offset": str(text.find(size)),
                "evidence_end_offset": str(text.find(size) + len(size)),
                "documentation_status": "supported",
                "manual_review_required": "True",
                "notes": "Heuristic extraction from TCGA report - requires manual review",
            })
        else:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "tumor_size",
                "gold_value": "",
                "evidence_text": "",
                "evidence_start_offset": "",
                "evidence_end_offset": "",
                "documentation_status": "not_documented",
                "manual_review_required": "True",
                "notes": "Size not found by heuristic - requires manual review",
            })

        # pT category
        pt = extract_pt_category(text)
        if pt:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "pathological_t_category",
                "gold_value": pt,
                "evidence_text": pt,
                "evidence_start_offset": str(text.find(pt)),
                "evidence_end_offset": str(text.find(pt) + len(pt)),
                "documentation_status": "supported",
                "manual_review_required": "True",
                "notes": "Heuristic extraction from TCGA report - requires manual review",
            })
        else:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "pathological_t_category",
                "gold_value": "",
                "evidence_text": "",
                "evidence_start_offset": "",
                "evidence_end_offset": "",
                "documentation_status": "not_documented",
                "manual_review_required": "True",
                "notes": "pT not found by heuristic - requires manual review",
            })

        # pN category
        pn = extract_pn_category(text)
        if pn:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "pathological_n_category",
                "gold_value": pn,
                "evidence_text": pn,
                "evidence_start_offset": str(text.find(pn)),
                "evidence_end_offset": str(text.find(pn) + len(pn)),
                "documentation_status": "supported",
                "manual_review_required": "True",
                "notes": "Heuristic extraction from TCGA report - requires manual review",
            })
        else:
            annotations.append({
                "report_id": report_id,
                "cancer_type": cancer_type,
                "variable_name": "pathological_n_category",
                "gold_value": "",
                "evidence_text": "",
                "evidence_start_offset": "",
                "evidence_end_offset": "",
                "documentation_status": "not_documented",
                "manual_review_required": "True",
                "notes": "pN not found by heuristic - requires manual review",
            })

    return annotations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare TCGA lung pathology reports."
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=0,
        help="Maximum number of reports to download (0 = default 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for reports JSON (default: data/raw/tcga_reports.json)",
    )
    parser.add_argument(
        "--annotations-output",
        type=Path,
        default=None,
        help="Output path for annotations CSV (default: data/raw/tcga_annotations.csv)",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=TCGA_LUNG_PROJECTS,
        help=f"TCGA project IDs (default: {' '.join(TCGA_LUNG_PROJECTS)})",
    )
    args = parser.parse_args()

    # Create output directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = args.output or RAW_DIR / "tcga_reports.json"
    annotations_path = args.annotations_output or RAW_DIR / "tcga_annotations.csv"

    # Query cases
    cases = query_gdc_cases(args.projects, args.max_reports)
    if not cases:
        print("No cases found.", file=sys.stderr)
        return 1

    # Download and process reports
    reports: list[TCGAReport] = []
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"\nProcessing {len(cases)} cases...")
    for i, case in enumerate(cases, 1):
        case_id = case.get("case_id", "")
        project_id = case.get("project", {}).get("project_id", "")

        if not case_id or not project_id:
            continue

        cancer_type = PROJECT_TO_CANCER_TYPE.get(project_id, "")
        if not cancer_type:
            continue

        print(f"\n[{i}/{len(cases)}] Processing case {case_id} ({cancer_type})...")

        # Get files for this case
        files = get_pathology_files(case_id)
        if not files:
            print(f"  No pathology files found for {case_id}")
            continue

        # Try to download and extract text from files
        text_parts = []
        for file_info in files[:3]:
            file_id = file_info.get("id", "")
            file_format = file_info.get("file_format", "")

            if not file_id:
                continue

            print(f"  Downloading file: {file_id} ({file_format})")
            content = download_file(file_id)

            if not content:
                continue

            if file_format == "pdf":
                text = extract_text_from_pdf(content)
            else:
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = ""

            if text and len(text) > 100:
                text_parts.append(text)

            time.sleep(0.5)

        if not text_parts:
            print(f"  No text extracted for {case_id}")
            continue

        full_text = "\n\n".join(text_parts)

        report = TCGAReport(
            case_id=case_id,
            project_id=project_id,
            cancer_type=cancer_type,
            report_id=f"TCGA-{case_id}",
            text=full_text[:50000],
            source_file=", ".join(
                f.get("file_name", "") for f in files[:3]
            ),
            download_date=download_date,
        )

        reports.append(report)
        print(f"  Extracted {len(full_text)} characters")

    if not reports:
        print("\nNo reports were successfully processed.", file=sys.stderr)
        return 1

    # Save reports
    print(f"\nSaving {len(reports)} reports to {output_path}...")
    reports_data = {
        "metadata": {
            "source": "TCGA",
            "projects": args.projects,
            "download_date": download_date,
            "total_reports": len(reports),
            "note": "These are real patient-derived reports from TCGA. Handle with care.",
        },
        "reports": [r.to_dict() for r in reports],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reports_data, f, indent=2, ensure_ascii=False)

    # Generate and save annotations
    print("Generating heuristic annotations...")
    annotations = generate_gold_annotations(reports)

    with open(annotations_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "report_id",
            "cancer_type",
            "variable_name",
            "gold_value",
            "evidence_text",
            "evidence_start_offset",
            "evidence_end_offset",
            "documentation_status",
            "manual_review_required",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotations)

    print(f"\nSummary:")
    print(f"  Reports saved: {output_path}")
    print(f"  Annotations saved: {annotations_path}")
    print(f"  Total reports: {len(reports)}")
    print(f"  Total annotations: {len(annotations)}")
    print(f"\nIMPORTANT: These annotations are heuristic and require manual review.")
    print("Do not use them as ground truth without expert validation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
