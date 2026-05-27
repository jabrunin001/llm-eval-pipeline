# Architecture Decision Records

## ADR-001: DuckDB as primary warehouse, Snowflake as portable target

**Status:** Accepted (2026-05-27)

**Context.** This is a portfolio project. It needs to be cloneable, runnable, and verifiable by any reviewer indefinitely. Snowflake's 30-day trial means the demo dies after a month. DuckDB is free forever, embedded, and zero-setup.

**Decision.** Use DuckDB as the default profile. Include a Snowflake profile in `profiles.yml` to demonstrate the dbt project is warehouse-portable. The Snowflake target is not in the critical path.

**Consequences.** Anyone can clone this repo, run `make all`, and see the dashboard in <10 minutes. The "I can ship this to a real warehouse" narrative is preserved in code, not just words.

---

## ADR-002: 3 Anthropic models × 3 runs each

**Status:** Accepted (2026-05-27)

**Context.** "Drift over time" requires multiple eval runs. Cost and time bound how many we can do. Wilson 95% CIs need n >= 2 to be meaningful.

**Decision.** Score 3 Anthropic models (Haiku 4.5, Sonnet 4.6, Opus 4.7) with 3 runs each at different sampling seeds. Total: 9 runs.

**Consequences.** Total API cost under $20 for a 200-question subset. Wilson CIs make the noise floor visible. Cross-model comparison is the dominant story; temporal drift is intentionally weak — flagged in README Limitations.

---

## ADR-003: Parsing lives in raw, not staging

**Status:** Accepted (2026-05-27)

**Context.** LLM completions are not always parseable into A/B/C/D. The parser is a regex stack. Bugs in the parser are likely; the cost of being wrong is having to re-call the API at $0.001-$0.015 per call.

**Decision.** Store `parsed_answer` and `is_correct` in `raw_eval_responses` alongside `raw_completion`. The raw completion is retained.

**Consequences.** "Re-parse historical runs with a better regex" is a SQL-only refresh — never an API rerun. The cost of parser experimentation is near-zero. Staging stays a pure projection.

---

## ADR-004: No judge-model scoring

**Status:** Accepted (2026-05-27)

**Context.** A more rigorous parser would use an LLM-as-judge to extract the answer from free-form completions. This handles refusals, equivocation, and "I think it's A but B is also reasonable" outputs.

**Decision.** Out of scope for this portfolio project. Use regex parsing. Track `unparseable_rate` as a first-class metric in `int_run_summary` so the regex's failures are visible.

**Consequences.** Some completions will be marked unparseable that a judge model could resolve. The Limitations section in the README calls this out explicitly. The shape of the pipeline supports adding a judge-model step later as a new column in `raw_eval_responses` — no schema rework needed.
