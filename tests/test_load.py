import json
from pathlib import Path
from unittest.mock import patch

from eval_pipeline.load import _hash_question_id, load_mmlu_to_warehouse
from eval_pipeline.warehouse import bootstrap_schema, connect

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
    rows = con.execute(
        "SELECT subject, answer FROM raw_mmlu_questions ORDER BY question_id"
    ).fetchall()
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


def test_load_filters_by_subject(tmp_path):
    db = tmp_path / "t.duckdb"
    con = connect(db)
    bootstrap_schema(con)
    sample = json.loads(FIXTURE.read_text())
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        # Caller filter is realised in load_mmlu_to_warehouse's `subjects` arg, which
        # is forwarded into _fetch_mmlu. We mock _fetch_mmlu to honour the filter.
        def fake_fetch(subjects):
            rows = [r for r in sample if not subjects or r["subject"] in subjects]
            return rows, "fixture-sha"
        m.side_effect = fake_fetch
        n = load_mmlu_to_warehouse(con, subjects=["high_school_physics"])
    assert n == 2
    subjects_loaded = {r[0] for r in con.execute(
        "SELECT subject FROM raw_mmlu_questions"
    ).fetchall()}
    assert subjects_loaded == {"high_school_physics"}


def test_load_normalizes_subject_to_lowercase(tmp_path):
    db = tmp_path / "t.duckdb"
    con = connect(db)
    bootstrap_schema(con)
    sample = [{"subject": "WORLD_HISTORY", "question": "q?",
               "choices": ["a", "b", "c", "d"], "answer": 0}]
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = (sample, "sha")
        load_mmlu_to_warehouse(con, subjects=None)
    subj = con.execute("SELECT subject FROM raw_mmlu_questions").fetchone()[0]
    assert subj == "world_history"


def test_load_maps_answer_index_to_letter_boundaries(tmp_path):
    db = tmp_path / "t.duckdb"
    con = connect(db)
    bootstrap_schema(con)
    sample = [
        {"subject": "s", "question": "q0", "choices": ["a", "b", "c", "d"], "answer": 0},
        {"subject": "s", "question": "q3", "choices": ["a", "b", "c", "d"], "answer": 3},
    ]
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = (sample, "sha")
        load_mmlu_to_warehouse(con, subjects=None)
    answers = {r[0] for r in con.execute(
        "SELECT answer FROM raw_mmlu_questions"
    ).fetchall()}
    assert answers == {"A", "D"}
