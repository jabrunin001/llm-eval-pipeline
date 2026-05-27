from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import typer
from dotenv import load_dotenv

from eval_pipeline.load import load_mmlu_to_warehouse
from eval_pipeline.models import Question
from eval_pipeline.prompts import PROMPT_TEMPLATE_VERSION
from eval_pipeline.runs import start_run
from eval_pipeline.score import score_run
from eval_pipeline.warehouse import bootstrap_schema, connect

load_dotenv()

app = typer.Typer(help="LLM eval pipeline CLI")

DEFAULT_SUBJECTS = [
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "high_school_mathematics",
    "world_history",
    "philosophy",
]


def _db_path() -> Path:
    return Path(os.getenv("EVAL_DB_PATH", "data/eval.duckdb"))


@app.command()
def load(
    subjects: list[str] | None = typer.Option(  # noqa: B008
        None, "--subject",
        help=(
            "Filter to subjects (repeat flag for multiple; "
            "defaults to 6 STEM/humanities subjects)"
        ),
    ),
) -> None:
    """Load MMLU into raw_mmlu_questions."""
    con = connect(_db_path())
    bootstrap_schema(con)
    n = load_mmlu_to_warehouse(con, subjects=subjects or DEFAULT_SUBJECTS)
    typer.echo(f"Loaded {n} questions")


def _score_impl(
    *,
    model: str,
    subset_size: int = 200,
    seed: int = 42,
    temperature: float = 0.0,
) -> None:
    """Internal scoring implementation (called by both `score` and `run` commands)."""
    from anthropic import AsyncAnthropic

    con = connect(_db_path())
    bootstrap_schema(con)
    rng = random.Random(seed)
    rows = con.execute(
        """SELECT question_id, subject, question, choice_a, choice_b, choice_c, choice_d,
                  answer, dataset_version
           FROM raw_mmlu_questions"""
    ).fetchall()
    if not rows:
        typer.echo(f"No questions loaded at {_db_path()}. Run `load` first.", err=True)
        raise typer.Exit(1)
    rng.shuffle(rows)
    rows = rows[:subset_size]
    questions = [
        Question(
            question_id=r[0], subject=r[1], question=r[2],
            choice_a=r[3], choice_b=r[4], choice_c=r[5], choice_d=r[6],
            answer=r[7], dataset_version=r[8],
        ) for r in rows
    ]
    rid = start_run(
        con, model=model, prompt_version=PROMPT_TEMPLATE_VERSION,
        seed=seed, subset_size=len(questions), temperature=temperature,
    )
    client = AsyncAnthropic()
    asyncio.run(score_run(con, client, rid, model, questions))
    typer.echo(f"Run complete: {rid}")


@app.command()
def score(
    model: str = typer.Option(..., "--model", help="Model id, e.g. claude-haiku-4-5-20251001"),
    subset_size: int = typer.Option(200, "--subset-size", help="Questions per run (default 200)"),
    seed: int = typer.Option(42, "--seed", help="Sampling seed for the question subset"),
    temperature: float = typer.Option(  # noqa: B008
        0.0, "--temperature", help="Scoring temperature (default 0.0)",
    ),
) -> None:
    """Score MMLU subset against a model."""
    _score_impl(model=model, subset_size=subset_size, seed=seed, temperature=temperature)


@app.command()
def run(
    models: list[str] = typer.Option(  # noqa: B008
        ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
        "--model", help="Models to score (repeat the flag for multiple)",
    ),
    n_runs: int = typer.Option(3, "--n-runs", help="Number of runs per model"),
    subset_size: int = typer.Option(200, "--subset-size", help="Questions per run"),
    base_seed: int = typer.Option(  # noqa: B008
        42, "--seed", help="Base sampling seed; per-run seed = base + run_index",
    ),
) -> None:
    """Run all models × n_runs in sequence."""
    for m in models:
        for i in range(n_runs):
            seed = base_seed + i
            typer.echo(f"--- {m} run {i+1}/{n_runs} (seed={seed}) ---")
            _score_impl(model=m, subset_size=subset_size, seed=seed, temperature=0.0)


if __name__ == "__main__":
    app()
