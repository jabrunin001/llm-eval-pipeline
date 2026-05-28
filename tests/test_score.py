from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval_pipeline.load import _hash_question_id
from eval_pipeline.models import Question
from eval_pipeline.runs import get_run, start_run
from eval_pipeline.score import score_run
from eval_pipeline.warehouse import bootstrap_schema, connect

FIXTURE = Path(__file__).parent / "fixtures" / "anthropic_responses.json"

def _seed_questions(con, n=3):
    qs = []
    for i in range(n):
        qid = _hash_question_id("math", f"q{i}")
        con.execute(
            """INSERT INTO raw_mmlu_questions
               VALUES (?, 'math', ?, 'a', 'b', 'c', 'd', 'B', 'sha', ?)""",
            [qid, f"q{i}", "2026-01-01"],
        )
        qs.append(Question(
            question_id=qid, subject="math", question=f"q{i}",
            choice_a="a", choice_b="b", choice_c="c", choice_d="d",
            answer="B", dataset_version="sha",
        ))
    return qs

def _fake_completion(text, in_tok=50, out_tok=1):
    m = MagicMock()
    m.content = [MagicMock(text=text)]
    m.usage = MagicMock(input_tokens=in_tok, output_tokens=out_tok)
    return m

@pytest.mark.asyncio
async def test_score_run_happy_path(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    bootstrap_schema(con)
    qs = _seed_questions(con, 3)
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=[
        _fake_completion("B"), _fake_completion("A"), _fake_completion("B"),
    ])
    rid = start_run(con, model="claude-haiku-4-5", prompt_version="v1",
                    seed=42, subset_size=3, temperature=0.0)
    await score_run(con, fake_client, rid, "claude-haiku-4-5", qs)
    rows = con.execute(
        "SELECT parsed_answer, is_correct FROM raw_eval_responses WHERE run_id = ?",
        [str(rid)],
    ).fetchall()
    assert len(rows) == 3
    assert {r[0] for r in rows} == {"A", "B"}
    assert get_run(con, rid).status == "completed"

@pytest.mark.asyncio
async def test_score_run_empty_completion_not_retried(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    bootstrap_schema(con)
    qs = _seed_questions(con, 1)
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_fake_completion("", out_tok=0))
    rid = start_run(con, model="m", prompt_version="v1", seed=1, subset_size=1, temperature=0.0)
    await score_run(con, fake_client, rid, "m", qs)
    row = con.execute(
        "SELECT parsed_answer, is_correct, api_error FROM raw_eval_responses WHERE run_id = ?",
        [str(rid)],
    ).fetchone()
    assert row[0] is None
    assert row[1] is None
    assert row[2] == "empty_completion"
    assert fake_client.messages.create.call_count == 1

@pytest.mark.asyncio
async def test_score_run_idempotency_duplicate_blocked(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    bootstrap_schema(con)
    qs = _seed_questions(con, 1)
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_fake_completion("B"))
    rid = start_run(con, model="m", prompt_version="v1", seed=1, subset_size=1, temperature=0.0)
    await score_run(con, fake_client, rid, "m", qs)
    # Re-score same questions for same run_id — should raise (UNIQUE violation)
    with pytest.raises(Exception, match=r"(?i)unique|duplicate|constraint"):
        await score_run(con, fake_client, rid, "m", qs)
    assert get_run(con, rid).status == "failed"


@pytest.mark.asyncio
async def test_score_run_marks_nonempty_unparseable(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    bootstrap_schema(con)
    qs = _seed_questions(con, 1)
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_fake_completion("I refuse to answer"))
    rid = start_run(con, model="m", prompt_version="v1", seed=1, subset_size=1, temperature=0.0)
    await score_run(con, fake_client, rid, "m", qs)
    row = con.execute(
        "SELECT parsed_answer, is_correct, api_error FROM raw_eval_responses WHERE run_id = ?",
        [str(rid)],
    ).fetchone()
    assert row[0] is None
    assert row[1] is None
    assert row[2] == "unparseable"
