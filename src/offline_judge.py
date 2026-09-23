"""Deterministic offline stand-in for the LLM judge that DeepEval metrics call.

Every DeepEval LLM-based metric is constructed with ``model=``. When given an
instance of :class:`deepeval.models.DeepEvalBaseLLM` that is not one of the
built-in provider models, the metric treats it as a "non-native" model
(``using_native_model=False``) and routes every judge call through::

    metric.model.generate_with_schema(prompt, schema=SchemaCls)      # sync
    metric.model.a_generate_with_schema(prompt, schema=SchemaCls)     # async

``DeepEvalBaseLLM`` implements both of those as thin wrappers that delegate to
``generate(prompt, schema=schema, ...)`` / ``a_generate(prompt, schema=schema,
...)``. The metric then JSON-parses the returned string
(``trimAndLoadJson``) and extracts the field its schema requires, e.g.
``data["verdicts"]``.

``OfflineJudgeModel`` implements exactly that seam: a canned, mode-driven JSON
document per (metric, schema) pair. ``mode="good"`` makes every judge verdict
favorable (scores -> 1.0 / 0.9), ``mode="bad"`` makes them unfavorable (scores
-> 0.0 / 0.2). No network is ever touched.
"""

import json
from typing import Optional, Type

from deepeval.models import DeepEvalBaseLLM


class OfflineJudgeModel(DeepEvalBaseLLM):
    """A :class:`DeepEvalBaseLLM` whose "judgements" are canned JSON.

    Args:
        mode: ``"good"`` returns favorable judge verdicts for every prompt;
            ``"bad"`` returns unfavorable ones. This is how the tests get a
            passing case and a failing case without any LLM.
    """

    def __init__(self, mode: str = "good"):
        if mode not in ("good", "bad"):
            raise ValueError("mode must be 'good' or 'bad'")
        self.mode = mode
        super().__init__(model="offline-judge")

    # -- DeepEvalBaseLLM interface -------------------------------------
    def load_model(self, *args, **kwargs) -> "OfflineJudgeModel":
        # Nothing to load: the judge is pure canned data.
        return self

    def get_model_name(self, *args, **kwargs) -> str:
        return "offline-judge"

    def generate(self, prompt, *args, schema: Optional[Type] = None, **kwargs) -> str:
        return self._respond(str(prompt), schema)

    async def a_generate(
        self, prompt, *args, schema: Optional[Type] = None, **kwargs
    ) -> str:
        return self._respond(str(prompt), schema)

    # -- canned responses ----------------------------------------------
    def _respond(self, prompt: str, schema: Optional[Type]) -> str:
        good = self.mode == "good"
        verdict = "yes" if good else "no"
        no_reason = {
            "faithfulness": "The claim contradicts the retrieval context.",
            "answer_relevancy": "The statement does not address the input.",
            "hallucination": "The output disagrees with the context.",
        }

        module = getattr(schema, "__module__", "") or ""
        name = getattr(schema, "__name__", "") or ""

        if module.startswith("deepeval.metrics.faithfulness"):
            if name == "Truths":
                return json.dumps(
                    {"truths": ["The retrieval context is the source of truth."]}
                )
            if name == "Claims":
                return json.dumps({"claims": ["The answer makes a factual claim."]})
            if name == "Verdicts":
                return json.dumps(
                    {
                        "verdicts": [
                            {
                                "verdict": verdict,
                                "reason": "" if good else no_reason["faithfulness"],
                            }
                        ]
                    }
                )
            if name == "FaithfulnessScoreReason":
                return json.dumps({"reason": "Offline stubbed reason."})

        if module.startswith("deepeval.metrics.answer_relevancy"):
            if name == "Statements":
                return json.dumps({"statements": ["A statement from the answer."]})
            if name == "Verdicts":
                return json.dumps(
                    {
                        "verdicts": [
                            {
                                "verdict": verdict,
                                "reason": ""
                                if good
                                else no_reason["answer_relevancy"],
                            }
                        ]
                    }
                )
            if name == "AnswerRelevancyScoreReason":
                return json.dumps({"reason": "Offline stubbed reason."})

        if module.startswith("deepeval.metrics.hallucination"):
            if name == "Verdicts":
                return json.dumps(
                    {
                        "verdicts": [
                            {
                                "verdict": verdict,
                                "reason": ""
                                if good
                                else no_reason["hallucination"],
                            }
                        ]
                    }
                )
            if name == "HallucinationScoreReason":
                return json.dumps({"reason": "Offline stubbed reason."})

        if module.startswith("deepeval.metrics.g_eval"):
            if name == "ReasonScore":
                # GEval normalizes a 0-10 judge score to 0-1.
                return json.dumps(
                    {"score": 9 if good else 2, "reason": "Offline stubbed reason."}
                )
            if name == "Steps":
                return json.dumps({"steps": ["Step 1", "Step 2"]})

        raise AssertionError(
            f"OfflineJudgeModel has no canned response for schema {module}.{name}"
        )
