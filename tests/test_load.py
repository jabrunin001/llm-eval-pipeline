import json
from pathlib import Path
from unittest.mock import patch
from eval_pipeline.load import load_mmlu_to_warehouse, _hash_question_id
from eval_pipeline.warehouse import connect, bootstrap_schema

FIXTURE = Path(__file__).parent / "fixtures" / "mmlu_sample.json"

def test_question_id_is_stable():
    a = _hash_question_id("math", "What is 2+2?")
    b = _hash_question_id("math", "What is 2+2?")
    c = _hash_question_id("math", "What is 3+3?")
    assert a == b
    assert a != c
    assert len(a) == 16  # short hex hash

def test_load_inserts_all_rows(tmp_path):
    db = tmp_path / "t.duckdb"
    con = connect(db)
    bootstrap_schema(con)
    sample = json.loads(FIXTURE.read_text())
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = (sample, "fixture-sha")
        n = load_mmlu_to_warehouse(con, subjects=None)
    assert n == 3
    rows = con.execute("SELECT subject, answer FROM raw_mmlu_questions ORDER BY question_id").fetchall()
    assert len(rows) == 3
    assert {r[1] for r in rows} <= {"A", "B", "C", "D"}

def test_load_is_idempotent(tmp_path):
    db = tmp_path / "t.duckdb"
    con = connect(db)
    bootstrap_schema(con)
    sample = json.loads(FIXTURE.read_text())
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = (sample, "fixture-sha")
        load_mmlu_to_warehouse(con, subjects=None)
        load_mmlu_to_warehouse(con, subjects=None)  # second call no-ops
    n = con.execute("SELECT COUNT(*) FROM raw_mmlu_questions").fetchone()[0]
    assert n == 3
