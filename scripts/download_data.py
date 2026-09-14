"""Fetch the EMSCAD / Kaggle "Real or Fake Job Posting" dataset.

The canonical file is `fake_job_postings.csv` (~18k rows, ~4% fraudulent). There
is no stable, auth-free public URL, so this script supports two paths:

  1. Kaggle API (preferred): requires `pip install kaggle` and a Kaggle API token
     at ~/.kaggle/kaggle.json. Downloads and unzips into data/.
  2. Manual: download from
       https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
     and drop `fake_job_postings.csv` into the data/ directory.

If neither is available, the training pipeline automatically falls back to the
synthetic generator, so the project still runs — this script only upgrades you to
the real benchmark.

Usage:
  python scripts/download_data.py            # try Kaggle API
  python scripts/download_data.py --check     # just report what's present
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobguard.config import settings  # noqa: E402

TARGET = settings.data_dir / "fake_job_postings.csv"
KAGGLE_DATASET = "shivamb/real-or-fake-fake-jobposting-prediction"


def _via_kaggle() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception:
        print("Kaggle API not installed (`pip install kaggle`). See manual steps below.")
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(KAGGLE_DATASET, path=str(settings.data_dir), unzip=True)
        return TARGET.exists()
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Only report presence.")
    args = ap.parse_args()

    if TARGET.exists():
        rows = sum(1 for _ in TARGET.open("r", encoding="utf-8", errors="replace")) - 1
        print(f"Dataset present: {TARGET} (~{rows} rows)")
        return 0

    if args.check:
        print(f"Dataset NOT present at {TARGET}. Run without --check to download, "
              "or the pipeline will use synthetic data.")
        return 0

    print("Attempting Kaggle download ...")
    if _via_kaggle():
        print(f"Downloaded -> {TARGET}")
        return 0

    print(
        "\nManual download instructions:\n"
        f"  1. Visit https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
        "  2. Download and unzip.\n"
        f"  3. Place `fake_job_postings.csv` in: {settings.data_dir}\n"
        "\nUntil then, training/inference will use the synthetic fallback dataset."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
