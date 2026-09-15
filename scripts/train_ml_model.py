#!/usr/bin/env python3
"""Train and save the local pathology ML extractor.

Usage:
    python scripts/train_ml_model.py
    python scripts/train_ml_model.py --output models/pathology_ml_extractor.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_model import DEFAULT_MODEL_PATH, PathologyMLExtractor, reset_ml_extractor_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the OncoExtract local ML extractor.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output joblib path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for logistic regression.")
    args = parser.parse_args()

    extractor = PathologyMLExtractor().train(seed=args.seed)
    path = extractor.save(args.output)
    reset_ml_extractor_cache()

    print(f"Saved ML extractor to {path}")
    print(f"Framework: {extractor.metadata.get('framework')}")
    print(f"Model name: {extractor.metadata.get('model_name')}")
    print(f"Reports used: {extractor.metadata.get('n_reports')}")
    for variable_name, counts in (extractor.metadata.get("label_counts") or {}).items():
        print(f"  {variable_name}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
