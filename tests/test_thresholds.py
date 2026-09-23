"""Threshold gate tests: good cases clear config/thresholds.yaml, bad cases fail it."""

import os

import yaml

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from src.custom_geval import build_root_cause_geval
from src.offline_judge import OfflineJudgeModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _thresholds():
    with open(os.path.join(REPO_ROOT, "config", "thresholds.yaml")) as f:
        return yaml.safe_load(f)


def _metrics():
    return {
        "faithfulness": (FaithfulnessMetric, {}),
        "answer_relevancy": (AnswerRelevancyMetric, {}),
        "hallucination": (HallucinationMetric, {}),
        "qa_root_cause_geval": (build_root_cause_geval, {}),
    }


def test_good_cases_meet_thresholds(good_case: LLMTestCase):
    thresholds = _thresholds()
    for name, (factory, kwargs) in _metrics().items():
        metric = factory(model=OfflineJudgeModel(mode="good"), **kwargs)
        score = metric.measure(good_case, _show_indicator=False)
        assert score >= thresholds[name], (
            f"{name}: good case scored {score}, below threshold {thresholds[name]}"
        )


def test_bad_cases_fail_thresholds(bad_case: LLMTestCase):
    thresholds = _thresholds()
    for name, (factory, kwargs) in _metrics().items():
        metric = factory(model=OfflineJudgeModel(mode="bad"), **kwargs)
        score = metric.measure(bad_case, _show_indicator=False)
        assert score < thresholds[name], (
            f"{name}: bad case scored {score}, did not fall below threshold {thresholds[name]}"
        )
