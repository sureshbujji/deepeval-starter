"""Offline DeepEval metric tests.

Each metric is evaluated with a stubbed judge (OfflineJudgeModel) instead of a
real LLM, so this suite runs in CI with no API key and no network. The "good"
case must score high, the "bad" case must score low — that spread is what makes
the suite a real regression gate for the metric wiring.
"""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from src.custom_geval import build_root_cause_geval
from src.offline_judge import OfflineJudgeModel


def _assert_good(score: float, metric_name: str):
    assert score >= 0.9, f"{metric_name}: good case scored {score}, expected >= 0.9"


def _assert_bad(score: float, metric_name: str):
    assert score <= 0.2, f"{metric_name}: bad case scored {score}, expected <= 0.2"


def test_faithfulness_good_case(good_case: LLMTestCase, good_judge: OfflineJudgeModel):
    metric = FaithfulnessMetric(model=good_judge)
    score = metric.measure(good_case, _show_indicator=False)
    _assert_good(score, "faithfulness")
    assert metric.is_successful()


def test_faithfulness_bad_case(bad_case: LLMTestCase, bad_judge: OfflineJudgeModel):
    metric = FaithfulnessMetric(model=bad_judge)
    score = metric.measure(bad_case, _show_indicator=False)
    _assert_bad(score, "faithfulness")
    assert not metric.is_successful()


def test_answer_relevancy_good_case(good_case: LLMTestCase, good_judge: OfflineJudgeModel):
    metric = AnswerRelevancyMetric(model=good_judge)
    score = metric.measure(good_case, _show_indicator=False)
    _assert_good(score, "answer_relevancy")
    assert metric.is_successful()


def test_answer_relevancy_bad_case(bad_case: LLMTestCase, bad_judge: OfflineJudgeModel):
    metric = AnswerRelevancyMetric(model=bad_judge)
    score = metric.measure(bad_case, _show_indicator=False)
    _assert_bad(score, "answer_relevancy")
    assert not metric.is_successful()


def test_hallucination_good_case(good_case: LLMTestCase, good_judge: OfflineJudgeModel):
    metric = HallucinationMetric(model=good_judge)
    score = metric.measure(good_case, _show_indicator=False)
    _assert_good(score, "hallucination")
    assert metric.is_successful()


def test_hallucination_bad_case(bad_case: LLMTestCase, bad_judge: OfflineJudgeModel):
    metric = HallucinationMetric(model=bad_judge)
    score = metric.measure(bad_case, _show_indicator=False)
    _assert_bad(score, "hallucination")
    assert not metric.is_successful()


def test_root_cause_geval_good_case(good_case: LLMTestCase):
    metric = build_root_cause_geval(model=OfflineJudgeModel(mode="good"))
    score = metric.measure(good_case, _show_indicator=False)
    _assert_good(score, "qa_root_cause_geval")
    assert metric.is_successful()


def test_root_cause_geval_bad_case(bad_case: LLMTestCase):
    metric = build_root_cause_geval(model=OfflineJudgeModel(mode="bad"))
    score = metric.measure(bad_case, _show_indicator=False)
    _assert_bad(score, "qa_root_cause_geval")
    assert not metric.is_successful()


def test_scores_are_deterministic(good_case: LLMTestCase, good_judge: OfflineJudgeModel):
    """The stubbed judge must return identical scores across runs."""
    first = FaithfulnessMetric(model=good_judge).measure(good_case, _show_indicator=False)
    second = FaithfulnessMetric(model=good_judge).measure(good_case, _show_indicator=False)
    assert first == second == 1.0
