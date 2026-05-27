# LLM Eval Pipeline — Design

**Date:** 2026-05-27
**Status:** Approved (brainstorming → implementation-plan)
**Purpose:** Portfolio artifact targeted at Anthropic analytics-engineering roles. Demonstrates AE chops applied to LLM evaluation data.

---

## 1. Goals & non-goals

### Goals
- Demonstrate analytics-engineering chops on eval data: clean schemas, lineage, freshness, idempotency, reproducible reruns.
- Be cloneable and runnable by any reviewer in <10 minutes with one API key.
- Survive as a portfolio link indefinitely (no expiring free trials in the critical path).
- Tell a clear story in the README that an Anthropic reviewer can grok in 90 seconds.

### Non-goals
- State-of-the-art benchmark numbers (MMLU is saturated; this is plumbing).
- Eval methodology innovation (no novel scoring, no judge-model setup, no contamination analysis).
- Production-scale (3 models × 3 runs is portfolio-scale; flag the gap honestly).
- Multi-provider scoring (single provider — Anthropic — for on-brand simplicity).

### Primary competency proved
**AE chops on eval data.** Clean datasets, lineage, drift legibility, reliability SLAs on the data feeding research decisions.

---

## 2. Stack

| layer | choice | rationale |
|---|---|---|
| Warehouse (default) | DuckDB | Free forever, embedded, zero-setup for reviewers, repo runs anywhere. |
| Warehouse (portable) | Snowflake (profile only) | Second dbt profile in `profiles.yml` to prove warehouse portability. Not required to run. |
| Transformation | dbt-core (with `dbt-duckdb`) | Industry-standard AE tool; lineage falls out of `dbt docs`. |
| Ingest + scoring | Python 3.11 + Anthropic SDK | `anthropic.AsyncAnthropic` for concurrent scoring. |
| Models scored | `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7` | Single provider, three capability tiers, comparative story. Pin to dated aliases at run time. |
| Eval dataset | MMLU subset (HuggingFace `cais/mmlu`) | 57 categories → category breakdown story; multiple-choice → trivial scoring. |
| Dashboard | Streamlit (local only) | `streamlit run dashboard/app.py`. No hosted demo (avoids hosting cost + rot). |
| CI | GitHub Actions | ruff + mypy + pytest + `dbt build` against ephemeral DuckDB. |
| Packaging | `pyproject.toml` + `uv` | Lockfile for reproducibility. |

---

## 3. Repository layout

```
llm-eval-pipeline/
├── README.md                       # narrative + why-this-matters
├── pyproject.toml                  # uv/pip-installable; pinned deps
├── uv.lock
├── Makefile                        # `make all`, `make seed`, `make score`, `make dashboard`
├── .github/workflows/ci.yml        # ruff, mypy, pytest, dbt build
├── .gitignore                      # data/, .env, dbt target/
├── eval_pipeline/                  # Python package
│   ├── __init__.py
│   ├── cli.py                      # `python -m eval_pipeline {load,score,run,resume}`
│   ├── load.py                     # HuggingFace → raw_mmlu_questions
│   ├── score.py                    # Anthropic API → raw_eval_responses (async)
│   ├── runs.py                     # writes raw_eval_runs metadata
│   ├── warehouse.py                # DuckDB connection helper (env-driven path)
│   ├── models.py                   # Pydantic types (Question, Response, RunMeta)
│   └── prompts.py                  # MMLU prompt template (frozen, versioned)
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml                # duckdb (default) + snowflake (portable)
│   ├── models/
│   │   ├── sources/sources.yml     # source defs + freshness
│   │   ├── staging/
│   │   │   ├── stg_mmlu_questions.sql
│   │   │   ├── stg_eval_runs.sql
│   │   │   └── stg_eval_responses.sql
│   │   ├── intermediate/
│   │   │   ├── int_scored_responses.sql
│   │   │   └── int_run_summary.sql
│   │   └── marts/
│   │       ├── mart_pass_rate_by_model.sql
│   │       ├── mart_category_breakdown.sql
│   │       └── mart_run_drift.sql
│   ├── seeds/
│   │   └── mmlu_subject_groups.csv # 57 subjects → ~6 groups (STEM, humanities, etc.)
│   ├── tests/
│   │   ├── assert_no_orphan_responses.sql
│   │   ├── assert_pass_rate_in_range.sql
│   │   └── assert_run_seq_dense.sql
│   └── macros/
│       └── wilson_ci.sql           # Wilson 95% confidence interval
├── dashboard/
│   └── app.py                      # Streamlit; reads marts via DuckDB
├── tests/                          # pytest for Python layer
│   ├── fixtures/
│   │   ├── mmlu_sample.json        # 10 questions
│   │   └── anthropic_responses.json
│   ├── test_parse_letter.py
│   ├── test_load.py
│   ├── test_score.py
│   ├── test_cli.py
│   └── test_warehouse.py
├── data/                           # gitignored
│   └── eval.duckdb
└── docs/
    ├── lineage.svg                 # exported from `dbt docs`
    ├── decisions.md                # 4 ADRs
    └── superpowers/
        └── specs/
            └── 2026-05-27-llm-eval-pipeline-design.md  # this file
```

---

## 4. End-to-end data flow

```
HuggingFace MMLU ──(load.py)──▶ raw_mmlu_questions ─┐
                                                     │
Anthropic API ───(score.py)──▶ raw_eval_responses ─┤
                                                     ├─▶ stg_* ─▶ int_* ─▶ mart_*
                              raw_eval_runs ────────┘                       │
                              (run_id, model, ts,                           │
                               seed, prompt_version)                        ▼
                                                                     Streamlit + dbt docs
```

**Key principle:** `raw_*` tables are immutable, append-only, populated only by Python. dbt reads them as `sources` and never writes back to them.

---

## 5. Raw schema (immutable layer)

### `raw_mmlu_questions` — static reference, loaded once

| column | type | notes |
|---|---|---|
| `question_id` | TEXT (PK) | stable hash of subject+question text |
| `subject` | TEXT | one of 57 MMLU categories |
| `question` | TEXT | prompt stem |
| `choice_a` … `choice_d` | TEXT | the four options |
| `answer` | CHAR(1) | ground truth letter A/B/C/D |
| `dataset_version` | TEXT | HuggingFace dataset revision SHA |
| `loaded_at` | TIMESTAMP | when ingest ran |

### `raw_eval_runs` — run ledger, one row per `score` invocation

| column | type | notes |
|---|---|---|
| `run_id` | UUID (PK) | generated at run start |
| `model` | TEXT | e.g., `claude-haiku-4-5-20251001` (dated alias for reproducibility) |
| `model_provider` | TEXT | `anthropic` |
| `prompt_version` | TEXT | git SHA of `prompts.py` at run time |
| `seed` | INT | sampling seed for question subset |
| `subset_size` | INT | how many questions sampled this run |
| `temperature` | FLOAT | scoring temperature (default 0.0) |
| `started_at` | TIMESTAMP | |
| `finished_at` | TIMESTAMP | |
| `status` | TEXT | `completed` / `failed` / `partial` |
| `error_message` | TEXT | nullable |

### `raw_eval_responses` — one row per (run, question) API call

| column | type | notes |
|---|---|---|
| `response_id` | UUID (PK) | |
| `run_id` | UUID (FK → runs) | unique constraint: `(run_id, question_id)` |
| `question_id` | TEXT (FK → questions) | |
| `raw_completion` | TEXT | full model output |
| `parsed_answer` | CHAR(1) | A/B/C/D or NULL if unparseable |
| `is_correct` | BOOLEAN | NULL if `parsed_answer IS NULL` |
| `latency_ms` | INT | |
| `input_tokens` | INT | for cost tracking |
| `output_tokens` | INT | |
| `api_error` | TEXT | nullable; populated if call failed |
| `responded_at` | TIMESTAMP | |

**Design choices flagged:**

- **`parsed_answer` lives in raw, not staging.** Parsing is non-trivial; storing it next to `raw_completion` makes "re-score with a better parser" a pure SQL refresh.
- **No surrogate keys in marts.** Joins from marts back to raw use natural keys (`run_id`, `question_id`) for debuggability.

---

## 6. dbt layer

### Staging (1:1 with raw, light cleanup)

- **`stg_mmlu_questions`** — type casts, lowercases `subject`.
  Tests: `unique(question_id)`, `not_null(answer)`, `accepted_values(answer, ['A','B','C','D'])`.
- **`stg_eval_runs`** — adds `duration_ms = finished_at - started_at`.
  Tests: `unique(run_id)`, `not_null(model)`, `accepted_values(status, ['completed','failed','partial'])`.
- **`stg_eval_responses`** — passes through; adds `is_correct_strict = coalesce(is_correct, false)` so downstream can distinguish "wrong" from "unparseable."
  Tests: `unique(response_id)`, `relationships(run_id → stg_eval_runs)`, `relationships(question_id → stg_mmlu_questions)`.

### Intermediate

- **`int_scored_responses`** — joins responses → runs → questions. Adds `subject_group` from `seeds/mmlu_subject_groups.csv`. One row per scored response.
- **`int_run_summary`** — per-run rollup: total questions, parsed, correct, pass rate, **unparseable rate**, mean latency, total cost. One row per `run_id`.

### Marts

- **`mart_pass_rate_by_model`** — one row per `(model, subject_group)`. Wilson 95% CI via `wilson_ci` macro. Columns: `model`, `subject_group`, `n_responses`, `pass_rate`, `ci_lower`, `ci_upper`, `n_runs`. **Headline table.**
- **`mart_category_breakdown`** — one row per `(model, subject)`. Finer-grained across all 57 subjects.
- **`mart_run_drift`** — one row per `(model, run_id)` ordered by `started_at`. Columns: `model`, `run_id`, `run_seq` (1, 2, 3 per model), `pass_rate`, `n_responses`, `delta_from_first_run`.

### Macros & sources

- `macros/wilson_ci.sql` — reusable Wilson 95% CI (~15 lines of SQL). Documented in README.
- `sources/sources.yml` — declares `raw_*` tables as dbt sources with freshness: `error_after: 14 days` on `raw_eval_runs.started_at`.

### Lineage

`dbt docs generate` produces the lineage graph. Exported to `docs/lineage.svg` via Makefile target.

---

## 7. Scoring (Python ingest layer)

### Single API call

```python
def score_one(client, model, question, prompt_version) -> ResponseRow:
    completion = client.messages.create(
        model=model,
        max_tokens=64,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": render(question)}],
    )
    raw = completion.content[0].text
    parsed = parse_letter(raw)
    return ResponseRow(
        response_id=uuid4(),
        run_id=...,
        question_id=question.question_id,
        raw_completion=raw,
        parsed_answer=parsed,
        is_correct=(parsed == question.answer) if parsed else None,
        latency_ms=...,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
        responded_at=now_utc(),
    )
```

### Run lifecycle

1. **Start**: insert into `raw_eval_runs` with `status='partial'`, capture git SHA + seed + subset_size.
2. **Sample** questions deterministically by `seed`.
3. **Score in async batches of N=10** via `anthropic.AsyncAnthropic` + `asyncio.Semaphore(10)`.
4. **Append rows after each batch** (not at end). Partial data survives process death.
5. **Finish**: update run to `status='completed'` (or `failed`/`partial`).

### Failure modes

| failure | response |
|---|---|
| `RateLimitError` | exponential backoff via `tenacity`, max 5 retries, jitter |
| `APIConnectionError` / 5xx | same retry policy |
| Empty completion / refusal | row written with `parsed_answer=NULL`, `api_error='empty_completion'`. **Not retried** — real eval signal. |
| Process killed mid-run | run stays `partial`; `resume <run_id>` fills missing `question_id`s. |
| Bad API key / quota | abort with `status='failed'`, clear CLI error. |

### Idempotency

- `raw_eval_responses` has UNIQUE constraint on `(run_id, question_id)` — duplicates error loudly.
- Every `score` invocation allocates a fresh `run_id`. No silent re-scoring.
- `resume` *fills* a partial run; it doesn't overwrite completed responses.

### Cost control

- `--subset-size` flag, default 200 questions per run.
- 3 models × 3 runs × 200 questions ≈ 1,800 calls. Estimated total cost <$20.
- `input_tokens`/`output_tokens` captured per response → `mart_cost_by_run` view (optional Phase-2 mart).

### CLI

```
python -m eval_pipeline load                                    # HuggingFace → DuckDB (one-shot)
python -m eval_pipeline score --model <model> --subset-size 200 --seed 42
python -m eval_pipeline run                                     # all 3 models × 3 seeds
python -m eval_pipeline resume <run_id>                         # finish a partial run
```

---

## 8. Testing

### Python tests (`tests/`)

| file | covers |
|---|---|
| `test_parse_letter.py` | regex parser vs. ~15 real model outputs + property test |
| `test_load.py` | mocked HuggingFace; verifies schema; stable `question_id` hashes |
| `test_score.py` | mocked `AsyncAnthropic`; happy path + rate-limit retry + empty completion + idempotency |
| `test_cli.py` | end-to-end with `--subset-size 5 --seed 42` against real API (gated by env var; skipped in CI) |
| `test_warehouse.py` | DuckDB connection helper, unique-constraint enforcement |

Fixtures: `mmlu_sample.json` (10 questions), `anthropic_responses.json` (10 recorded API responses). Both committed.

### dbt tests

Declared inline in Section 6 (uniqueness, not-null, accepted-values, relationships). Plus three singular tests:

- `assert_no_orphan_responses.sql` — every response's `run_id` exists in runs.
- `assert_pass_rate_in_range.sql` — every mart `pass_rate` ∈ [0, 1].
- `assert_run_seq_dense.sql` — `mart_run_drift.run_seq` has no gaps per model.

---

## 9. CI (`.github/workflows/ci.yml`)

Three parallel jobs:

```yaml
jobs:
  python:
    - uv sync --frozen
    - ruff check .
    - mypy eval_pipeline
    - pytest -m "not integration"

  dbt:
    - uv sync --frozen
    - python -m eval_pipeline load --target ci   # fixture-populated CI DuckDB
    - dbt seed --profiles-dir dbt --target ci
    - dbt build --profiles-dir dbt --target ci

  docs:
    - dbt docs generate --profiles-dir dbt --target ci
    - upload docs/ as artifact
```

The `ci` dbt target uses ephemeral DuckDB + fixture data so dbt runs in <30s without API keys.

**No coverage gate.** Named tests cover the failure modes that matter.

---

## 10. README — front door

Structure:

```
# LLM Eval Pipeline

[ 1-paragraph what-this-is ]
[ animated GIF of Streamlit dashboard, 4-5s ]

## Why eval-data plumbing matters for an AI lab        ← ~120-word hook
## What this pipeline does                              ← lineage SVG + 3 bullets
## What's interesting in the code                       ← AE differentiators (linked to files)
## Running it yourself                                  ← copy-pasteable
## Decisions (ADRs)                                     ← link to docs/decisions.md
## Limitations & what I'd do with a real eval team      ← earnest gaps section
```

### "What's interesting" — the four differentiators

- **Parsing separated from scoring.** `parsed_answer` in `raw_eval_responses` alongside `raw_completion`. Bad regex → SQL refresh, not API rerun. (`eval_pipeline/score.py`)
- **`unparseable_rate` as first-class metric.** Distinguishes "wrong" from "couldn't parse." (`dbt/models/intermediate/int_run_summary.sql`)
- **Wilson 95% CI on pass rates.** Three runs per model isn't a lot — the CI math makes the noise floor visible. (`dbt/macros/wilson_ci.sql`)
- **Run ledger + idempotent inserts.** UUID per run, captured prompt-version git SHA, UNIQUE on `(run_id, question_id)`. (`eval_pipeline/runs.py`)

### Limitations section

Honest list:
- Three runs/model is barely enough for CI to be meaningful — real teams run hundreds.
- MMLU is saturated; useful here as a known quantity, not SOTA signal.
- Regex parser; a judge-model parser would handle long-form refusals.
- Drift across hours, not weeks — real drift detection wants cron history.

### `docs/decisions.md` — 4 ADRs

1. Why DuckDB primary (portability + zero-cost re-runs); how Snowflake portability is preserved.
2. Why 3 Anthropic models, 3 runs each (cost + statistical floor).
3. Why parsing lives in raw, not staging.
4. Why no judge-model scoring (scope; flagged in Limitations).

---

## 11. Out of scope (explicit)

- Hosted dashboard (Streamlit Cloud / Hex).
- Scheduled eval via GitHub Actions cron.
- Cross-provider scoring (OpenAI, Ollama, etc.).
- Judge-model parsing for free-form completions.
- Contamination analysis.
- Cost-by-run dashboard panel (mart_cost_by_run noted as optional Phase 2).
- Snowflake as primary warehouse (profile only; not in critical path).

---

## 12. Open questions

None at design time. Reopen if:
- Anthropic pricing changes materially during build (cost ceiling at $50 — switch to Haiku-only if exceeded).
- HuggingFace `cais/mmlu` schema changes (pin `dataset_version` and document the SHA).

---

## 13. Estimated effort

**1-2 weekends, in this order:**

1. Project scaffold + DuckDB + `load` (HuggingFace → `raw_mmlu_questions`).
2. `score` async loop + run ledger + tests.
3. dbt staging + intermediate + marts + macro.
4. Streamlit dashboard reading marts.
5. CI workflow + dbt CI fixtures.
6. README + ADRs + lineage SVG + final polish.

Each step is ~half a day. Items 5-6 are where ~30% of the portfolio value lives; don't truncate them.
