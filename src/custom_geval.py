"""A QA-flavored custom G-Eval metric wired to the offline judge.

G-Eval (LLM-as-judge with explicit criteria) is DeepEval's escape hatch for
things canned metrics can't express. Here the criterion is a classic QA
question: *did the answer actually identify the root cause?* — scored by the
same deterministic :class:`OfflineJudgeModel` as everything else, so it runs
offline in CI.
"""

from typing import Optional, Union

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams

from src.offline_judge import OfflineJudgeModel

CRITERION = "Does the answer correctly identify the root cause of the issue?"

EVALUATION_STEPS = [
    "Read the input and identify what the reported issue is.",
    "Read the retrieval context and identify the documented root cause.",
    "Check whether the actual output names that same root cause "
    "(paraphrase is fine; the technical cause must match).",
    "Penalize answers that blame a different component or are vague.",
]

EVALUATION_PARAMS = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
    SingleTurnParams.RETRIEVAL_CONTEXT,
]


def build_root_cause_geval(
    model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    threshold: float = 0.5,
) -> GEval:
    """Build the root-cause G-Eval metric.

    Args:
        model: judge model; defaults to a "good"-mode
            :class:`OfflineJudgeModel`. Pass ``OfflineJudgeModel(mode="bad")``
            to simulate a failing answer.
        threshold: minimum passing score.
    """
    return GEval(
        name="qa-root-cause",
        evaluation_params=EVALUATION_PARAMS,
        criteria=CRITERION,
        evaluation_steps=EVALUATION_STEPS,
        model=model or OfflineJudgeModel(mode="good"),
        threshold=threshold,
    )
