"""CI quality gate: score the sample cases and enforce config/thresholds.yaml.

Good sample case must meet every threshold; bad sample case must fail every
threshold. Exits non-zero on any violation so CI fails loudly.

Usage:
    python src/check_thresholds.py [--data-dir data] [--thresholds config/thresholds.yaml]
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import yaml  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402

from src.custom_geval import build_root_cause_geval  # noqa: E402
from src.offline_judge import OfflineJudgeModel  # noqa: E402


def load_case(path: str) -> LLMTestCase:
    with open(path) as f:
        d = json.load(f)
    return LLMTestCase(
        input=d["input"],
        actual_output=d["actual_output"],
        expected_output=d["expected_output"],
        retrieval_context=d["retrieval_context"],
        context=d["retrieval_context"],  # HallucinationMetric reads `context`
    )


def build_metrics():
    good = OfflineJudgeModel(mode="good")
    bad = OfflineJudgeModel(mode="bad")
    return {
        "faithfulness": (FaithfulnessMetric(model=good), FaithfulnessMetric(model=bad)),
        "answer_relevancy": (
            AnswerRelevancyMetric(model=good),
            AnswerRelevancyMetric(model=bad),
        ),
        "hallucination": (HallucinationMetric(model=good), HallucinationMetric(model=bad)),
        "qa_root_cause_geval": (
            build_root_cause_geval(model=OfflineJudgeModel(mode="good")),
            build_root_cause_geval(model=OfflineJudgeModel(mode="bad")),
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument(
        "--thresholds", default=os.path.join(REPO_ROOT, "config", "thresholds.yaml")
    )
    args = ap.parse_args()

    with open(args.thresholds) as f:
        thresholds = yaml.safe_load(f)

    good_case = load_case(os.path.join(args.data_dir, "good_case.json"))
    bad_case = load_case(os.path.join(args.data_dir, "bad_case.json"))

    failures = []
    print(f"{'metric':<22} {'good':>6} {'bad':>6} {'threshold':>9}  gate")
    print("-" * 56)
    for name, (good_m, bad_m) in build_metrics().items():
        threshold = thresholds[name]
        good_score = good_m.measure(good_case, _show_indicator=False)
        bad_score = bad_m.measure(bad_case, _show_indicator=False)
        good_ok = good_score >= threshold
        bad_ok = bad_score < threshold
        status = "PASS" if (good_ok and bad_ok) else "FAIL"
        print(
            f"{name:<22} {good_score:>6.2f} {bad_score:>6.2f} {threshold:>9.2f}  {status}"
        )
        if not good_ok:
            failures.append(f"{name}: good case {good_score:.2f} < threshold {threshold}")
        if not bad_ok:
            failures.append(f"{name}: bad case {bad_score:.2f} >= threshold {threshold}")

    if failures:
        print("\nThreshold gate FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nThreshold gate PASSED: good cases clear all thresholds, bad cases fail them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
