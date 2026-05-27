# LLM Eval Pipeline

A reproducible analytics-engineering pipeline for LLM evaluation data — MMLU benchmark scored against three Claude models, modeled in dbt, served as a local Streamlit dashboard. Built to demonstrate the kind of plumbing an eval team needs from analytics engineering.

> ⚠️ Portfolio project. Three runs per model is not enough for real eval conclusions — see [Limitations](#limitations).

![dashboard screenshot placeholder](docs/dashboard.png)

## Why eval-data plumbing matters for an AI lab

Every public benchmark number you see — MMLU 87%, HumanEval 92% — sits on top of a pipeline that loaded a dataset, sampled it, ran a model against it, parsed completions, joined to ground truth, and aggregated. That pipeline rarely shows up in papers. When it breaks subtly — a prompt-template typo, a parser regex that drops 3% of completions as "unparseable", a dataset-version drift — the benchmark number lies and a research decision gets made on a lie.

Analytics engineering for an eval team means owning that pipeline like it's a financial reporting system: versioned schemas, lineage, freshness checks, separated parsing from scoring, reproducible reruns. This repo is what that looks like for a small MMLU subset.

## What this pipeline does

```
HuggingFace MMLU ──(load.py)──▶ raw_mmlu_questions ─┐
                                                     │
Anthropic API ───(score.py)──▶ raw_eval_responses ─┤
                                                     ├─▶ stg_* ─▶ int_* ─▶ mart_*
                              raw_eval_runs ────────┘
```

- **3 Claude models** (Haiku 4.5, Sonnet 4.6, Opus 4.7), **3 runs each**, scored against an MMLU subset.
- **dbt project** with `staging → intermediate → marts`. Three marts: `mart_pass_rate_by_model` (with Wilson 95% CIs), `mart_category_breakdown` (per-subject), `mart_run_drift`.
- **Streamlit dashboard** for exploration.
- **dbt docs lineage** at `docs/lineage.svg`.

## What's interesting in the code

- **Parsing is separated from scoring.** `parsed_answer` lives in `raw_eval_responses` alongside `raw_completion`. A bad regex doesn't require re-calling the API — it's a pure SQL refresh. → [`eval_pipeline/score.py`](eval_pipeline/score.py)
- **`unparseable_rate` is a first-class metric.** Honest evals distinguish "model got it wrong" from "parser couldn't tell." → [`dbt/models/intermediate/int_run_summary.sql`](dbt/models/intermediate/int_run_summary.sql)
- **Wilson 95% confidence intervals on pass rates.** Three runs per model isn't a lot — the CI math makes the noise floor visible. → [`dbt/macros/wilson_ci.sql`](dbt/macros/wilson_ci.sql)
- **Run ledger + idempotent inserts.** Every run gets a UUID, a captured prompt-template version + git SHA, and a UNIQUE constraint on `(run_id, question_id)`. Drift is legible because reruns can't silently merge. → [`eval_pipeline/runs.py`](eval_pipeline/runs.py)

## Running it yourself

Requirements: Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), an Anthropic API key.

```bash
git clone <this-repo>
cd llm-eval-pipeline
cp .env.example .env  # then edit ANTHROPIC_API_KEY
uv sync

# 1. Load MMLU into DuckDB (one-time, ~30s)
uv run python -m eval_pipeline.cli load

# 2. Score all 3 models × 3 runs (~5 min, ~$5-$20 in API)
uv run python -m eval_pipeline.cli run

# 3. Build the dbt project
cd dbt && uv run dbt seed --profiles-dir . && uv run dbt build --profiles-dir .

# 4. Launch the dashboard
cd .. && uv run streamlit run dashboard/app.py
```

Or: `make all`.

## Decisions

See [`docs/decisions.md`](docs/decisions.md) for 4 ADRs covering warehouse choice, model setup, parsing boundary, and judge-model scope.

## Limitations

Earnestness as signal:

- **Three runs per model is barely enough for the Wilson CI to be meaningful.** A real eval team runs hundreds. The CI bands in the dashboard are honest about this.
- **MMLU is a saturated benchmark.** It's used here as a known quantity for tooling, not as a state-of-the-art signal. Pass rates in the 80-90% range are expected; differences between models on a 200-question subset may be inside the CI.
- **The parser is regex-based.** A judge-model parser would handle long-form refusals and equivocation better. `unparseable_rate` is tracked so the regex's failures are visible.
- **Drift is shown across runs hours apart, not weeks.** Real drift detection wants cron history.

## License

MIT.
