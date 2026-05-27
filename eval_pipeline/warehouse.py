from pathlib import Path
import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_mmlu_questions (
    question_id     VARCHAR PRIMARY KEY,
    subject         VARCHAR NOT NULL,
    question        VARCHAR NOT NULL,
    choice_a        VARCHAR NOT NULL,
    choice_b        VARCHAR NOT NULL,
    choice_c        VARCHAR NOT NULL,
    choice_d        VARCHAR NOT NULL,
    answer          CHAR(1) NOT NULL CHECK (answer IN ('A','B','C','D')),
    dataset_version VARCHAR NOT NULL,
    loaded_at       TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_eval_runs (
    run_id          VARCHAR PRIMARY KEY,
    model           VARCHAR NOT NULL,
    model_provider  VARCHAR NOT NULL,
    prompt_version  VARCHAR NOT NULL,
    seed            INTEGER NOT NULL,
    subset_size     INTEGER NOT NULL,
    temperature     DOUBLE NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    status          VARCHAR NOT NULL CHECK (status IN ('completed','failed','partial')),
    error_message   VARCHAR
);

CREATE TABLE IF NOT EXISTS raw_eval_responses (
    response_id     VARCHAR PRIMARY KEY,
    run_id          VARCHAR NOT NULL REFERENCES raw_eval_runs(run_id),
    question_id     VARCHAR NOT NULL REFERENCES raw_mmlu_questions(question_id),
    raw_completion  VARCHAR NOT NULL,
    parsed_answer   CHAR(1) CHECK (parsed_answer IN ('A','B','C','D')),
    is_correct      BOOLEAN,
    latency_ms      BIGINT NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    api_error       VARCHAR,
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
