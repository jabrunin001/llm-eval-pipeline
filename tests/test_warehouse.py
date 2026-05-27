import pytest
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from eval_pipeline.warehouse import connect, bootstrap_schema

def test_bootstrap_creates_three_raw_tables(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    bootstrap_schema(con)
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()}
    assert {"raw_mmlu_questions", "raw_eval_runs", "raw_eval_responses"}.issubset(tables)

def test_unique_constraint_on_run_question(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    bootstrap_schema(con)
    rid = str(uuid4())
    qid = "q1"
    # Need a question row first because raw_eval_responses FK-references raw_mmlu_questions
    con.execute("""
        INSERT INTO raw_mmlu_questions VALUES (?, 'math', 'q?', 'a', 'b', 'c', 'd', 'A', 'sha', CURRENT_TIMESTAMP)
    """, [qid])
    con.execute("""
        INSERT INTO raw_eval_runs VALUES (?, 'm', 'anthropic', 'sha', 42, 1, 0.0, ?, NULL, 'partial', NULL)
    """, [rid, datetime.now(timezone.utc)])
    con.execute("""
        INSERT INTO raw_eval_responses VALUES (?, ?, ?, 'raw', 'A', true, 10, 1, 1, NULL, ?)
    """, [str(uuid4()), rid, qid, datetime.now(timezone.utc)])
    with pytest.raises(Exception, match=r"(?i)unique|duplicate|constraint"):
        con.execute("""
            INSERT INTO raw_eval_responses VALUES (?, ?, ?, 'raw2', 'B', false, 10, 1, 1, NULL, ?)
        """, [str(uuid4()), rid, qid, datetime.now(timezone.utc)])

def test_bootstrap_is_idempotent(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    bootstrap_schema(con)
    bootstrap_schema(con)  # second call must not raise
