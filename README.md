# deepeval-starter

I built this to answer one question the way a QA lead actually asks it: *"Can I
trust this LLM feature enough to ship it, and can I prove it in CI?"*

This is a starter project showing how I'd adopt [DeepEval](https://github.com/confident-ai/deepeval)
— the open-source LLM evaluation framework — on a QA team: real DeepEval
metrics (`FaithfulnessMetric`, `AnswerRelevancyMetric`, `HallucinationMetric`,
plus a custom G-Eval), golden sample cases, and per-metric pass thresholds
enforced as a CI gate. Everything runs **100% offline**: the LLM judge every
metric calls under the hood is replaced by a deterministic stub, so `pytest`
passes with no API key and no network.

## What DeepEval is (and when I'd use it vs. a hand-rolled harness)

DeepEval is an LLM evaluation framework: you define `LLMTestCase`s (input,
actual output, expected output, retrieval context), run ready-made metrics over
them, and get scores + pass/fail per metric. Under the hood the LLM-based
metrics call a judge model to extract claims, render verdicts, and score them.

The alternative is a **hand-rolled LLM-as-judge harness**: your own prompt
templates, your own JSON parsing, a golden dataset in JSON/CSV, and your own CI
gate script. I've built those too. I'd reach for the hand-rolled harness when:

- the evaluation logic is simple and bespoke (e.g. exact-match on structured
  outputs, regex checks on tool calls),
- I want zero framework dependencies in a latency-sensitive pipeline,
- the team needs full control of every prompt token and retry.

I'd reach for **DeepEval** when:

- I need *standard, defensible* metrics (faithfulness, answer relevancy,
  hallucination, G-Eval) that the rest of the industry recognizes,
- the team wants metric implementations maintained by someone else (verdict
  parsing, scoring edge cases, multimodal support),
- I want a path to the hosted Confident AI platform later (tracing, datasets,
  regression dashboards) without rewriting the tests.

This repo is the on-ramp: it proves the wiring (metrics + thresholds + CI)
before you spend a dollar on judge-model API calls.

## How it fits in a QA pipeline

```
                          ┌─────────────────────────┐
                          │   LLM feature under     │
                          │   test (your app/API)   │
                          └────────────┬────────────┘
                                       │ produces
                                       ▼
┌──────────────┐   ┌─────────────────────────────────────────┐
│ Golden       │   │  DeepEval evaluation (this repo)        │
│ dataset      │──▶│                                         │
│ data/*.json  │   │  LLMTestCase(input, actual_output,      │
└──────────────┘   │    expected_output, retrieval_context)  │
                   │        │                                │
                   │        ▼                                │
                   │  FaithfulnessMetric ──┐                 │
                   │  AnswerRelevancyMetric ├─▶ judge model   │
                   │  HallucinationMetric ──┤   (stubbed      │
                   │  GEval "qa-root-cause" ┘   offline)     │
                   │        │                                │
                   │        ▼                                │
                   │  config/thresholds.yaml ── CI gate ──▶  │
                   └─────────────────────────────────────────┘
                                   │ pass / fail
                                   ▼
                          ┌─────────────────┐
                          │  merge / deploy │
                          │  decision       │
                          └─────────────────┘
```

In production the "judge model" box is a real LLM (swap the stub for
`deepeval`'s OpenAI/Anthropic/etc. models). In CI — here — it's the offline
stub, so the gate validates *your metric configuration and thresholds*, not a
live model's mood.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the metric tests (no API key needed)
pytest -v

# run the threshold gate
python src/check_thresholds.py
```

Sample output:

```
$ pytest -v
tests/test_metrics_offline.py::test_faithfulness_good_case PASSED
tests/test_metrics_offline.py::test_faithfulness_bad_case PASSED
tests/test_metrics_offline.py::test_answer_relevancy_good_case PASSED
tests/test_metrics_offline.py::test_answer_relevancy_bad_case PASSED
tests/test_metrics_offline.py::test_hallucination_good_case PASSED
tests/test_metrics_offline.py::test_hallucination_bad_case PASSED
tests/test_metrics_offline.py::test_root_cause_geval_good_case PASSED
tests/test_metrics_offline.py::test_root_cause_geval_bad_case PASSED
tests/test_metrics_offline.py::test_scores_are_deterministic PASSED
tests/test_thresholds.py::test_good_cases_meet_thresholds PASSED
tests/test_thresholds.py::test_bad_cases_fail_thresholds PASSED
11 passed

$ python src/check_thresholds.py
metric                   good    bad threshold  gate
--------------------------------------------------------
faithfulness             1.00   0.00      0.70  PASS
answer_relevancy         1.00   0.00      0.70  PASS
hallucination            1.00   0.00      0.70  PASS
qa_root_cause_geval      0.90   0.20      0.50  PASS

Threshold gate PASSED: good cases clear all thresholds, bad cases fail them.
```

## How offline mode works (the precise version)

DeepEval metrics don't score text themselves — they call a judge model. In
deepeval 4.2.5 the call path is:

1. You pass a model to the metric: `FaithfulnessMetric(model=my_model)`.
2. `deepeval.metrics.utils.models.initialize_model()` sees `my_model` is an
   instance of `deepeval.models.DeepEvalBaseLLM` but not a built-in provider
   model, so it keeps it as-is with `using_native_model=False`.
3. Every judge call goes through `metric.model.generate_with_schema(prompt,
   schema=SchemaCls)` (sync `measure`) or `metric.model.a_generate_with_schema(...)`
   (async). `DeepEvalBaseLLM` implements both as thin wrappers delegating to
   `generate(prompt, schema=schema, ...)` / `a_generate(prompt, schema=schema, ...)`.
4. The metric JSON-parses the returned string (`trimAndLoadJson`) and extracts
   the field its schema demands — e.g. `data["verdicts"]` for the QAG verdict
   loop, `data["score"]` for G-Eval.

**The stub** (`src/offline_judge.py`, class `OfflineJudgeModel`) implements
that exact `DeepEvalBaseLLM` interface: `load_model()` (returns self —
nothing to load), `get_model_name()` (returns `"offline-judge"`),
`generate(prompt, schema=None, ...)` and `a_generate(prompt, schema=None,
...)`. The response is dispatched on the **schema's module + class name** —
e.g. `deepeval.metrics.faithfulness.schema.Verdicts` — and returns a canned
JSON document with the fields that metric parses:

| metric | schema → canned response |
|---|---|
| Faithfulness | `Truths`→`{"truths": [...]}`, `Claims`→`{"claims": [...]}`, `Verdicts`→`{"verdicts": [{"verdict": "yes"/"no", ...}]}`, `FaithfulnessScoreReason`→`{"reason": ...}` |
| AnswerRelevancy | `Statements`→`{"statements": [...]}`, `Verdicts`→`{"verdicts": [...]}`, `AnswerRelevancyScoreReason`→`{"reason": ...}` |
| Hallucination | `Verdicts`→`{"verdicts": [...]}`, `HallucinationScoreReason`→`{"reason": ...}` |
| GEval | `ReasonScore`→`{"score": 9 / 2, "reason": ...}` (GEval normalizes the 0–10 judge score to 0–1, so 9→0.9, 2→0.2) |

**Why scores are deterministic:** the stub is mode-driven, not prompt-driven.
`OfflineJudgeModel(mode="good")` returns favorable verdicts (`"yes"`, score 9)
for every prompt; `mode="bad"` returns unfavorable ones (`"no"`, score 2). It
does not parse the test case or pretend to judge — the good/bad distinction
comes from which fixture the test wires up. Scores follow from DeepEval's own
scoring code: `score_qag_verdicts` = fraction of verdicts in the passing bucket
(YES/BORDERLINE for faithfulness & relevancy, YES for hallucination). That
gives good=1.0 / bad=0.0 on the three QAG metrics and 0.9/0.2 on G-Eval,
run after run. This validates metric wiring, threshold config, and CI — swap
in a real judge model when you want real judgments.

**Telemetry:** DeepEval ships PostHog telemetry. `DEEPEVAL_TELEMETRY_OPT_OUT=1`
(declared in `.env.example`, set in `tests/conftest.py` before any deepeval
import, and in `.github/workflows/ci.yml`) routes it to a no-op backend —
verified via `deepeval/telemetry/client.py::telemetry_opt_out()`.

## Project layout

```
deepeval-starter/
├── src/
│   ├── offline_judge.py      # OfflineJudgeModel: deterministic DeepEvalBaseLLM stub
│   ├── custom_geval.py       # build_root_cause_geval(): QA-flavored G-Eval metric
│   └── check_thresholds.py   # CI gate: scores samples, enforces thresholds.yaml
├── tests/
│   ├── conftest.py           # telemetry opt-out + good/bad fixtures
│   ├── test_metrics_offline.py
│   └── test_thresholds.py
├── data/
│   ├── good_case.json        # (input, expected, actual, context): grounded answer
│   └── bad_case.json         # same input/context, hallucinated wrong root cause
├── config/thresholds.yaml    # per-metric pass thresholds
├── .github/workflows/ci.yml # install → pytest → threshold gate, no secrets
└── requirements.txt          # deepeval==4.2.5, pytest==9.1.1, pyyaml==6.0.3
```

## Roadmap

- [ ] Add golden-dataset growth: mine real incident postmortems into `data/`.
- [ ] Swap the stub for a real judge (`OPENAI_API_KEY`) behind an env flag, keep
      the stub as the default CI path.
- [ ] Add `ContextualRecallMetric` / `ContextualPrecisionMetric` for RAG answers.
- [ ] Publish per-run scores as CI artifacts (JSON) and trend them.
- [ ] Evaluate pushing runs to Confident AI for a regression dashboard.

## License

MIT — see [LICENSE](LICENSE).
