from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from uuid import UUID, uuid4

import duckdb

from eval_pipeline.models import RunMeta


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def start_run(
    con: duckdb.DuckDBPyConnection,
    *,
    model: str,
    prompt_version: str,
    seed: int,
    subset_size: int,
    temperature: float,
    model_provider: str = "anthropic",
) -> UUID:
    rid = uuid4()
    full_prompt_version = f"{prompt_version}@{_git_sha()}"
    con.execute(
        """INSERT INTO raw_eval_runs
           (run_id, model, model_provider, prompt_version, seed, subset_size,
            temperature, started_at, finished_at, status, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'partial', NULL)""",
        [str(rid), model, model_provider, full_prompt_version, seed,
         subset_size, temperature, datetime.now(UTC)],
    )
    return rid


def finish_run(
    con: duckdb.DuckDBPyConnection,
    run_id: UUID,
    *,
    status: str = "completed",
) -> None:
    con.execute(
        """UPDATE raw_eval_runs SET status = ?, finished_at = ? WHERE run_id = ?""",
        [status, datetime.now(UTC), str(run_id)],
    )


def mark_run_failed(
    con: duckdb.DuckDBPyConnection,
    run_id: UUID,
    error_message: str,
) -> None:
    con.execute(
        """UPDATE raw_eval_runs
           SET status = 'failed', finished_at = ?, error_message = ?
           WHERE run_id = ?""",
        [datetime.now(UTC), error_message, str(run_id)],
    )


def get_run(con: duckdb.DuckDBPyConnection, run_id: UUID) -> RunMeta:
    row = con.execute(
        """SELECT run_id, model, model_provider, prompt_version, seed, subset_size,
                  temperature, started_at, finished_at, status, error_message
           FROM raw_eval_runs WHERE run_id = ?""",
        [str(run_id)],
    ).fetchone()
    if row is None:
        raise LookupError(f"run not found: {run_id}")
    return RunMeta(
        run_id=UUID(str(row[0])),
        model=row[1],
        model_provider=row[2],
        prompt_version=row[3],
        seed=row[4],
        subset_size=row[5],
        temperature=row[6],
        started_at=row[7],
        finished_at=row[8],
        status=row[9],
        error_message=row[10],
    )
