from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import duckdb
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from eval_pipeline.models import Question, ResponseRow
from eval_pipeline.prompts import SYSTEM_PROMPT, parse_letter, render_prompt
from eval_pipeline.runs import finish_run, mark_run_failed

CONCURRENCY = 10
MAX_TOKENS = 64


class _Retryable(Exception):
    pass


@retry(
    retry=retry_if_exception_type(_Retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _call_with_retry(client: Any, model: str, q: Question) -> Any:
    try:
        return await client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": render_prompt(q)}],
        )
    except Exception as e:
        cls = type(e).__name__
        if any(k in cls for k in ("RateLimit", "APIConnection", "APIStatus", "InternalServer")):
            raise _Retryable(str(e)) from e
        raise


async def _score_one(client: Any, model: str, q: Question, run_id: UUID) -> ResponseRow:
    started = time.perf_counter()
    try:
        completion = await _call_with_retry(client, model, q)
        raw = completion.content[0].text if completion.content else ""
        parsed = parse_letter(raw)
        is_correct = (parsed == q.answer) if parsed else None
        if not raw.strip():
            api_error = "empty_completion"
        elif parsed is None:
            api_error = "unparseable"
        else:
            api_error = None
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ResponseRow(
            response_id=uuid4(),
            run_id=run_id,
            question_id=q.question_id,
            raw_completion=raw,
            parsed_answer=parsed,
            is_correct=is_correct,
            latency_ms=latency_ms,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            api_error=api_error,
            responded_at=datetime.now(UTC),
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ResponseRow(
            response_id=uuid4(),
            run_id=run_id,
            question_id=q.question_id,
            raw_completion="",
            parsed_answer=None,
            is_correct=None,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            api_error=f"{type(e).__name__}: {str(e)[:200]}",
            responded_at=datetime.now(UTC),
        )


def _insert_responses(con: duckdb.DuckDBPyConnection, rows: list[ResponseRow]) -> None:
    for r in rows:
        con.execute(
            """INSERT INTO raw_eval_responses VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [
                str(r.response_id),
                str(r.run_id),
                r.question_id,
                r.raw_completion,
                r.parsed_answer,
                r.is_correct,
                r.latency_ms,
                r.input_tokens,
                r.output_tokens,
                r.api_error,
                r.responded_at,
            ],
        )


async def score_run(
    con: duckdb.DuckDBPyConnection,
    client: Any,
    run_id: UUID,
    model: str,
    questions: list[Question],
    batch_size: int = CONCURRENCY,
) -> None:
    sem = asyncio.Semaphore(batch_size)

    async def bounded(q: Question) -> ResponseRow:
        async with sem:
            return await _score_one(client, model, q, run_id)

    try:
        results = await asyncio.gather(*[bounded(q) for q in questions])
        _insert_responses(con, results)
        finish_run(con, run_id, status="completed")
    except Exception as e:
        mark_run_failed(con, run_id, f"{type(e).__name__}: {e}")
        raise
