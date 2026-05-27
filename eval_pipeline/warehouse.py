from pathlib import Path
import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_mmlu_questions (
    question_id     TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    question        TEXT NOT NULL,
    choice_a        TEXT NOT NULL,
    choice_b        TEXT NOT NULL,
    choice_c        TEXT NOT NULL,
    choice_d        TEXT NOT NULL,
    answer          CHAR(1) NOT NULL CHECK (answer IN ('A','B','C','D')),
    dataset_version TEXT NOT NULL,
    loaded_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_eval_runs (
    run_id          UUID PRIMARY KEY,
    model           TEXT NOT NULL,
    model_provider  TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    subset_size     INTEGER NOT NULL,
    temperature     DOUBLE NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    status          TEXT NOT NULL CHECK (status IN ('completed','failed','partial')),
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS raw_eval_responses (
    response_id     UUID PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES raw_eval_runs(run_id),
    question_id     TEXT NOT NULL REFERENCES raw_mmlu_questions(question_id),
    raw_completion  TEXT NOT NULL,
    parsed_answer   CHAR(1) CHECK (parsed_answer IN ('A','B','C','D')),
    is_correct      BOOLEAN,
    latency_ms      INTEGER NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    api_error       TEXT,
    responded_at    TIMESTAMP NOT NULL,
    UNIQUE (run_id, question_id)
);
"""

def connect(db_path: Path | str) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))

def bootstrap_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)
