"""Pytest fixtures for offline DeepEval runs.

No API keys, no network: every metric is driven by OfflineJudgeModel, and
telemetry is opted out via DEEPEVAL_TELEMETRY_OPT_OUT (set here before
deepeval is imported, and also in CI).
"""

import json
import os
import sys

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from deepeval.test_case import LLMTestCase  # noqa: E402

from src.offline_judge import OfflineJudgeModel  # noqa: E402


def _load_case(name: str) -> LLMTestCase:
    path = os.path.join(REPO_ROOT, "data", f"{name}.json")
    with open(path) as f:
        d = json.load(f)
    return LLMTestCase(
        input=d["input"],
        actual_output=d["actual_output"],
        expected_output=d["expected_output"],
        retrieval_context=d["retrieval_context"],
        context=d["retrieval_context"],  # HallucinationMetric reads `context`
    )


@pytest.fixture()
def good_case() -> LLMTestCase:
    return _load_case("good_case")


@pytest.fixture()
def bad_case() -> LLMTestCase:
    return _load_case("bad_case")


@pytest.fixture()
def good_judge() -> OfflineJudgeModel:
    return OfflineJudgeModel(mode="good")


@pytest.fixture()
def bad_judge() -> OfflineJudgeModel:
    return OfflineJudgeModel(mode="bad")
