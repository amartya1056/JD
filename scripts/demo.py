"""Stage-2 verification: run the inference pipeline on the seed samples.

    python scripts/demo.py
    python scripts/demo.py path/to/posting.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobguard.parser import parse_raw_text  # noqa: E402
from jobguard.pipeline import InferencePipeline  # noqa: E402


def analyze_file(pipeline: InferencePipeline, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    posting = parse_raw_text(text)
    result = pipeline.analyze(posting)
    print("\n" + "#" * 72)
    print(f"# {path.name}")
    print("#" * 72)
    print(f"VERDICT      : {result.verdict}")
    print(f"Confidence   : {result.confidence:.0%}")
    print(f"Risk score   : {result.risk_score}/100")
    print(f"P(real)={result.probability_real:.3f}  P(fake)={result.probability_fake:.3f}")
    print(f"Branch probs : {result.branch_scores}")
    print("Flagged reasons:")
    for r in result.flagged_reasons or ["(none)"]:
        print(f"   - {r}")
    print("Top suspicious phrases:", result.explanations["top_suspicious_phrases"] or "(none)")
    print("SHAP top features:")
    for f in result.explanations["shap_top_features"]:
        print(f"   - {f['label']}: {f['direction']} (impact {f['impact']})")
    print(f"Advice: {result.advice}")


def main() -> int:
    pipeline = InferencePipeline.load()
    samples_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        files = sorted(samples_dir.glob("*.txt"))
    for f in files:
        analyze_file(pipeline, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
