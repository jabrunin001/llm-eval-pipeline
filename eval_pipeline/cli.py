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
    subjects: list[str] | None = typer.Option(None, "--subject", help="Filter to subjects"),  # noqa: B008
) -> None:
    """Load MMLU into raw_mmlu_questions."""
    con = connect(_db_path())
    bootstrap_schema(con)
    n = load_mmlu_to_warehouse(con, subjects=subjects or DEFAULT_SUBJECTS)
    typer.echo(f"Loaded {n} questions")


@app.command()
def score(
    model: str = typer.Option(..., "--model", help="Model id, e.g. claude-haiku-4-5-20251001"),  # noqa: B008
    subset_size: int = typer.Option(200, "--subset-size"),  # noqa: B008
    seed: int = typer.Option(42, "--seed"),  # noqa: B008
    temperature: float = typer.Option(0.0, "--temperature"),  # noqa: B008
) -> None:
    """Score MMLU subset against a model."""
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
        typer.echo("No questions loaded. Run `load` first.", err=True)
        raise typer.Exit(1)
    rng.shuffle(rows)
    rows = rows[:subset_size]
    questions = [
        Question(
            question_id=r[0],
            subject=r[1],
            question=r[2],
            choice_a=r[3],
            choice_b=r[4],
            choice_c=r[5],
            choice_d=r[6],
            answer=r[7],
            dataset_version=r[8],
        )
        for r in rows
    ]
    rid = start_run(
        con,
        model=model,
        prompt_version=PROMPT_TEMPLATE_VERSION,
        seed=seed,
        subset_size=len(questions),
        temperature=temperature,
    )
    client = AsyncAnthropic()
    asyncio.run(score_run(con, client, rid, model, questions))
    typer.echo(f"Run complete: {rid}")


@app.command()
def run(
    models: list[str] = typer.Option(  # noqa: B008
        ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
        "--model",
        help="Models to score (can repeat)",
    ),
    n_runs: int = typer.Option(3, "--n-runs"),  # noqa: B008
    subset_size: int = typer.Option(200, "--subset-size"),  # noqa: B008
    base_seed: int = typer.Option(42, "--seed"),  # noqa: B008
) -> None:
    """Run all models x n_runs."""
    for m in models:
        for i in range(n_runs):
            seed = base_seed + i
            typer.echo(f"--- {m} run {i + 1}/{n_runs} (seed={seed}) ---")
            score(model=m, subset_size=subset_size, seed=seed, temperature=0.0)


if __name__ == "__main__":
    app()
