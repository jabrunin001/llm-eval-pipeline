# LLM Eval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible LLM eval pipeline that ingests MMLU into DuckDB, scores it against 3 Claude models with idempotent run tracking, models it in dbt with Wilson CIs, and serves it via a local Streamlit dashboard. Targets Anthropic analytics-engineering interview portfolio.

**Architecture:** Two-stage pipeline. Python `eval_pipeline` package handles ingest + async scoring → writes to immutable `raw_*` tables in DuckDB. dbt project transforms `raw → staging → intermediate → marts`. Streamlit reads marts via DuckDB connection. CI runs ruff/mypy/pytest + `dbt build` against fixture-populated ephemeral DuckDB.

**Tech Stack:** Python 3.11, `uv`, `anthropic` SDK (async), `pydantic`, `tenacity`, `duckdb`, `dbt-core` + `dbt-duckdb`, `streamlit`, `pytest`, GitHub Actions.

---

## File Structure (locked at plan time)

```
llm-eval-pipeline/
├── README.md                                       # Task 21
├── pyproject.toml                                  # Task 1
├── uv.lock                                         # Task 1
├── .gitignore                                      # Task 1
├── .env.example                                    # Task 1
├── Makefile                                        # Task 22
├── .github/workflows/ci.yml                        # Task 19
├── eval_pipeline/
│   ├── __init__.py                                 # Task 1
│   ├── models.py                                   # Task 2
│   ├── warehouse.py                                # Task 3
│   ├── prompts.py                                  # Task 5
│   ├── load.py                                     # Task 4
│   ├── runs.py                                     # Task 6
│   ├── score.py                                    # Task 7
│   └── cli.py                                      # Task 8
├── dbt/
│   ├── dbt_project.yml                             # Task 9
│   ├── profiles.yml                                # Task 9
│   ├── models/
│   │   ├── sources/sources.yml                     # Task 10
│   │   ├── staging/{stg_mmlu_questions,stg_eval_runs,stg_eval_responses}.{sql,yml}  # Tasks 10-11
│   │   ├── intermediate/{int_scored_responses,int_run_summary}.sql  # Task 13
│   │   └── marts/{mart_pass_rate_by_model,mart_category_breakdown,mart_run_drift}.{sql,yml}  # Tasks 15-16
│   ├── macros/wilson_ci.sql                        # Task 14
│   ├── seeds/mmlu_subject_groups.csv               # Task 12
│   └── tests/{assert_no_orphan_responses,assert_pass_rate_in_range,assert_run_seq_dense}.sql  # Task 17
├── dashboard/app.py                                # Task 18
├── tests/
│   ├── fixtures/{mmlu_sample.json,anthropic_responses.json}  # Tasks 4, 7
│   ├── test_warehouse.py                           # Task 3
│   ├── test_load.py                                # Task 4
│   ├── test_parse_letter.py                        # Task 5
│   ├── test_runs.py                                # Task 6
│   ├── test_score.py                               # Task 7
│   └── test_cli.py                                 # Task 8
└── docs/
    ├── decisions.md                                # Task 20
    ├── lineage.svg                                 # Task 22 (generated)
    ├── superpowers/specs/2026-05-27-llm-eval-pipeline-design.md  # already exists
    └── superpowers/plans/2026-05-27-llm-eval-pipeline.md         # this file
```

---

## Task 1: Project scaffold + dependencies

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `eval_pipeline/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "eval-pipeline"
version = "0.1.0"
description = "LLM eval pipeline: MMLU vs Claude models, modeled in dbt"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "datasets>=3.0.0",
    "duckdb>=1.1.0",
    "pydantic>=2.9.0",
    "tenacity>=9.0.0",
    "typer>=0.13.0",
    "python-dotenv>=1.0.0",
    "dbt-core>=1.8.0",
    "dbt-duckdb>=1.8.0",
    "streamlit>=1.40.0",
    "pandas>=2.2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
    "respx>=0.21.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]

[tool.mypy]
strict = true
python_version = "3.11"
exclude = ["dbt/", "tests/fixtures/"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["integration: live API tests, skipped in CI"]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
.venv/
.uv-cache/
.mypy_cache/
.ruff_cache/
.pytest_cache/
*.egg-info/

# Data + secrets
data/
*.duckdb
*.duckdb.wal
.env

# dbt
dbt/target/
dbt/dbt_packages/
dbt/logs/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Create `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
EVAL_DB_PATH=data/eval.duckdb
```

- [ ] **Step 4: Create `eval_pipeline/__init__.py`**

```python
"""LLM eval pipeline: MMLU vs Claude models."""
__version__ = "0.1.0"
```

- [ ] **Step 5: Lock dependencies**

Run: `uv sync`
Expected: `uv.lock` created, `.venv/` populated, no errors.

- [ ] **Step 6: Verify package imports**

Run: `uv run python -c "import eval_pipeline; print(eval_pipeline.__version__)"`
Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example eval_pipeline/__init__.py
git commit -m "feat: project scaffold with pinned dependencies"
```

---

## Task 2: Pydantic data models

**Files:**
- Create: `eval_pipeline/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing test `tests/test_models.py`**

```python
import pytest
from datetime import datetime, timezone
from uuid import uuid4
from eval_pipeline.models import Question, ResponseRow, RunMeta

def test_question_validates_answer_letter():
    q = Question(
        question_id="abc", subject="math", question="2+2?",
        choice_a="3", choice_b="4", choice_c="5", choice_d="6",
        answer="B", dataset_version="sha123",
    )
    assert q.answer == "B"

def test_question_rejects_invalid_answer():
    with pytest.raises(ValueError):
        Question(
            question_id="abc", subject="math", question="2+2?",
            choice_a="3", choice_b="4", choice_c="5", choice_d="6",
            answer="X", dataset_version="sha",
        )

def test_response_row_allows_null_parsed_answer():
    r = ResponseRow(
        response_id=uuid4(), run_id=uuid4(), question_id="abc",
        raw_completion="garbage", parsed_answer=None, is_correct=None,
        latency_ms=100, input_tokens=10, output_tokens=5,
        api_error=None, responded_at=datetime.now(timezone.utc),
    )
    assert r.parsed_answer is None
    assert r.is_correct is None

def test_run_meta_status_enum():
    with pytest.raises(ValueError):
        RunMeta(
            run_id=uuid4(), model="claude-haiku-4-5", model_provider="anthropic",
            prompt_version="sha", seed=42, subset_size=10, temperature=0.0,
            started_at=datetime.now(timezone.utc), finished_at=None,
            status="bogus", error_message=None,
        )
```

- [ ] **Step 2: Run test — should fail (no models.py yet)**

Run: `uv run pytest tests/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'eval_pipeline.models'`

- [ ] **Step 3: Create `eval_pipeline/models.py`**

```python
from datetime import datetime
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field

AnswerLetter = Literal["A", "B", "C", "D"]
RunStatus = Literal["completed", "failed", "partial"]

class Question(BaseModel):
    question_id: str
    subject: str
    question: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    answer: AnswerLetter
    dataset_version: str
    loaded_at: datetime | None = None

class ResponseRow(BaseModel):
    response_id: UUID
    run_id: UUID
    question_id: str
    raw_completion: str
    parsed_answer: AnswerLetter | None
    is_correct: bool | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    api_error: str | None
    responded_at: datetime

class RunMeta(BaseModel):
    run_id: UUID
    model: str
    model_provider: str
    prompt_version: str
    seed: int
    subset_size: int
    temperature: float = Field(ge=0.0, le=2.0)
    started_at: datetime
    finished_at: datetime | None
    status: RunStatus
    error_message: str | None
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add eval_pipeline/models.py tests/test_models.py
git commit -m "feat: pydantic models for Question, ResponseRow, RunMeta"
```

---

## Task 3: Warehouse (DuckDB schema bootstrap)

**Files:**
- Create: `eval_pipeline/warehouse.py`
- Test: `tests/test_warehouse.py`

- [ ] **Step 1: Write failing test `tests/test_warehouse.py`**

```python
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
    # Insert minimal run so FK doesn't trip if enforced
    con.execute("""
        INSERT INTO raw_eval_runs VALUES (?, 'm', 'anthropic', 'sha', 42, 1, 0.0, ?, NULL, 'partial', NULL)
    """, [rid, datetime.now(timezone.utc)])
    con.execute("""
        INSERT INTO raw_eval_responses VALUES (?, ?, 'q1', 'raw', 'A', true, 10, 1, 1, NULL, ?)
    """, [str(uuid4()), rid, datetime.now(timezone.utc)])
    with pytest.raises(Exception, match=r"(?i)unique|duplicate|constraint"):
        con.execute("""
            INSERT INTO raw_eval_responses VALUES (?, ?, 'q1', 'raw2', 'B', false, 10, 1, 1, NULL, ?)
        """, [str(uuid4()), rid, datetime.now(timezone.utc)])

def test_bootstrap_is_idempotent(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    bootstrap_schema(con)
    bootstrap_schema(con)  # second call must not raise
```

- [ ] **Step 2: Run test — should fail**

Run: `uv run pytest tests/test_warehouse.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `eval_pipeline/warehouse.py`**

```python
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
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/test_warehouse.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add eval_pipeline/warehouse.py tests/test_warehouse.py
git commit -m "feat: DuckDB schema bootstrap with FK + unique constraints"
```

---

## Task 4: Load MMLU from HuggingFace

**Files:**
- Create: `eval_pipeline/load.py`, `tests/fixtures/mmlu_sample.json`
- Test: `tests/test_load.py`

- [ ] **Step 1: Create fixture `tests/fixtures/mmlu_sample.json`**

```json
[
  {"subject": "high_school_physics", "question": "What is the SI unit of force?", "choices": ["Joule", "Newton", "Watt", "Pascal"], "answer": 1},
  {"subject": "high_school_physics", "question": "Speed of light in vacuum (m/s)?", "choices": ["3e6", "3e8", "3e10", "3e5"], "answer": 1},
  {"subject": "world_history", "question": "Year the Berlin Wall fell?", "choices": ["1987", "1988", "1989", "1990"], "answer": 2}
]
```

- [ ] **Step 2: Write failing test `tests/test_load.py`**

```python
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
```

- [ ] **Step 3: Run test — should fail**

Run: `uv run pytest tests/test_load.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 4: Create `eval_pipeline/load.py`**

```python
from __future__ import annotations
import hashlib
import duckdb
from datetime import datetime, timezone

_LETTERS = ["A", "B", "C", "D"]

def _hash_question_id(subject: str, question: str) -> str:
    return hashlib.sha256(f"{subject}|{question}".encode()).hexdigest()[:16]

def _fetch_mmlu(subjects: list[str] | None) -> tuple[list[dict], str]:
    """Fetch MMLU from HuggingFace. Returns (rows, dataset_version_sha).

    Real implementation uses `datasets.load_dataset('cais/mmlu', 'all', split='test')`.
    Mocked in tests.
    """
    from datasets import load_dataset  # lazy import
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for r in ds:
        if subjects and r["subject"] not in subjects:
            continue
        rows.append({
            "subject": r["subject"],
            "question": r["question"],
            "choices": r["choices"],
            "answer": r["answer"],
        })
    version_sha = getattr(ds, "_fingerprint", "unknown")[:16]
    return rows, version_sha

def load_mmlu_to_warehouse(
    con: duckdb.DuckDBPyConnection,
    subjects: list[str] | None = None,
) -> int:
    rows, version = _fetch_mmlu(subjects)
    now = datetime.now(timezone.utc)
    inserted = 0
    for r in rows:
        qid = _hash_question_id(r["subject"], r["question"])
        choices = r["choices"]
        answer_letter = _LETTERS[r["answer"]]
        try:
            con.execute(
                """INSERT INTO raw_mmlu_questions
                   (question_id, subject, question, choice_a, choice_b, choice_c, choice_d,
                    answer, dataset_version, loaded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [qid, r["subject"].lower(), r["question"],
                 choices[0], choices[1], choices[2], choices[3],
                 answer_letter, version, now],
            )
            inserted += 1
        except duckdb.ConstraintException:
            continue  # already loaded; idempotent
    return inserted
```

- [ ] **Step 5: Run tests — should pass**

Run: `uv run pytest tests/test_load.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add eval_pipeline/load.py tests/test_load.py tests/fixtures/mmlu_sample.json
git commit -m "feat: load MMLU from HuggingFace into raw_mmlu_questions"
```

---

## Task 5: Prompts + answer parsing

**Files:**
- Create: `eval_pipeline/prompts.py`
- Test: `tests/test_parse_letter.py`

- [ ] **Step 1: Write failing test `tests/test_parse_letter.py`**

```python
import pytest
from eval_pipeline.prompts import parse_letter, render_prompt, SYSTEM_PROMPT, PROMPT_TEMPLATE_VERSION
from eval_pipeline.models import Question

@pytest.mark.parametrize("raw,expected", [
    ("A", "A"),
    ("B.", "B"),
    ("C)", "C"),
    ("(D)", "D"),
    ("The answer is C.", "C"),
    ("The answer is **B**.", "B"),
    ("I think it's A", "A"),
    ("Let me think... B", "B"),
    ("The correct answer is (D).", "D"),
    ("answer: A", "A"),
    ("**Answer: C**", "C"),
    ("c)", "C"),  # case-insensitive
    ("", None),
    ("I refuse to answer.", None),
    ("E", None),  # out of range
    ("The answer is 4", None),  # numeric
])
def test_parse_letter(raw, expected):
    assert parse_letter(raw) == expected

def test_render_prompt_contains_choices():
    q = Question(
        question_id="x", subject="math", question="2+2?",
        choice_a="3", choice_b="4", choice_c="5", choice_d="6",
        answer="B", dataset_version="sha",
    )
    rendered = render_prompt(q)
    assert "2+2?" in rendered
    assert "A) 3" in rendered
    assert "D) 6" in rendered

def test_prompt_template_version_is_set():
    assert PROMPT_TEMPLATE_VERSION  # non-empty
    assert SYSTEM_PROMPT
```

- [ ] **Step 2: Run test — should fail**

Run: `uv run pytest tests/test_parse_letter.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `eval_pipeline/prompts.py`**

```python
import re
from eval_pipeline.models import Question

PROMPT_TEMPLATE_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam. For each question, respond with "
    "ONLY the single letter (A, B, C, or D) of the correct answer. "
    "Do not explain, do not add punctuation, just the letter."
)

USER_TEMPLATE = """Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""

def render_prompt(q: Question) -> str:
    return USER_TEMPLATE.format(
        question=q.question, a=q.choice_a, b=q.choice_b, c=q.choice_c, d=q.choice_d
    )

_LETTER_PATTERNS = [
    re.compile(r"\bAnswer\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\banswer is\s*\*?\*?\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\*\*([A-D])\*\*"),
    re.compile(r"\(([A-D])\)", re.IGNORECASE),
    re.compile(r"^\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\b([A-D])\s*[\.\)]", re.IGNORECASE),
    re.compile(r"\b([A-D])\b", re.IGNORECASE),  # last resort: any standalone letter
]

def parse_letter(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    for pat in _LETTER_PATTERNS:
        m = pat.search(raw)
        if m:
            return m.group(1).upper()
    return None
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/test_parse_letter.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add eval_pipeline/prompts.py tests/test_parse_letter.py
git commit -m "feat: prompt template and answer-letter parser"
```

---

## Task 6: Run ledger (`runs.py`)

**Files:**
- Create: `eval_pipeline/runs.py`
- Test: `tests/test_runs.py`

- [ ] **Step 1: Write failing test `tests/test_runs.py`**

```python
from datetime import datetime, timezone
from eval_pipeline.runs import start_run, finish_run, mark_run_failed, get_run
from eval_pipeline.warehouse import connect, bootstrap_schema

def test_start_run_inserts_partial(tmp_path):
    con = connect(tmp_path / "t.duckdb"); bootstrap_schema(con)
    rid = start_run(con, model="claude-haiku-4-5", prompt_version="sha1",
                    seed=42, subset_size=10, temperature=0.0)
    run = get_run(con, rid)
    assert run.status == "partial"
    assert run.finished_at is None
    assert run.model == "claude-haiku-4-5"

def test_finish_run_completes(tmp_path):
    con = connect(tmp_path / "t.duckdb"); bootstrap_schema(con)
    rid = start_run(con, model="m", prompt_version="s", seed=1, subset_size=1, temperature=0.0)
    finish_run(con, rid, status="completed")
    run = get_run(con, rid)
    assert run.status == "completed"
    assert run.finished_at is not None

def test_mark_run_failed_records_error(tmp_path):
    con = connect(tmp_path / "t.duckdb"); bootstrap_schema(con)
    rid = start_run(con, model="m", prompt_version="s", seed=1, subset_size=1, temperature=0.0)
    mark_run_failed(con, rid, "API quota exceeded")
    run = get_run(con, rid)
    assert run.status == "failed"
    assert run.error_message == "API quota exceeded"
```

- [ ] **Step 2: Run test — should fail**

Run: `uv run pytest tests/test_runs.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `eval_pipeline/runs.py`**

```python
from __future__ import annotations
import subprocess
from datetime import datetime, timezone
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
         subset_size, temperature, datetime.now(timezone.utc)],
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
        [status, datetime.now(timezone.utc), str(run_id)],
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
        [datetime.now(timezone.utc), error_message, str(run_id)],
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
        run_id=UUID(str(row[0])), model=row[1], model_provider=row[2],
        prompt_version=row[3], seed=row[4], subset_size=row[5], temperature=row[6],
        started_at=row[7], finished_at=row[8], status=row[9], error_message=row[10],
    )
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/test_runs.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add eval_pipeline/runs.py tests/test_runs.py
git commit -m "feat: run ledger with start/finish/fail lifecycle"
```

---

## Task 7: Async scoring (`score.py`)

**Files:**
- Create: `eval_pipeline/score.py`, `tests/fixtures/anthropic_responses.json`
- Test: `tests/test_score.py`

- [ ] **Step 1: Create fixture `tests/fixtures/anthropic_responses.json`**

```json
[
  {"completion": "B", "input_tokens": 50, "output_tokens": 1},
  {"completion": "The answer is B.", "input_tokens": 50, "output_tokens": 6},
  {"completion": "C", "input_tokens": 50, "output_tokens": 1},
  {"completion": "", "input_tokens": 50, "output_tokens": 0},
  {"completion": "I cannot answer this.", "input_tokens": 50, "output_tokens": 6}
]
```

- [ ] **Step 2: Write failing test `tests/test_score.py`**

```python
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest
from eval_pipeline.models import Question
from eval_pipeline.score import score_run
from eval_pipeline.warehouse import connect, bootstrap_schema
from eval_pipeline.load import _hash_question_id
from eval_pipeline.runs import start_run, get_run

FIXTURE = Path(__file__).parent / "fixtures" / "anthropic_responses.json"

def _seed_questions(con, n=3):
    qs = []
    for i in range(n):
        qid = _hash_question_id("math", f"q{i}")
        con.execute(
            """INSERT INTO raw_mmlu_questions VALUES (?, 'math', ?, 'a', 'b', 'c', 'd', 'B', 'sha', CURRENT_TIMESTAMP)""",
            [qid, f"q{i}"],
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
    con = connect(tmp_path / "t.duckdb"); bootstrap_schema(con)
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
    con = connect(tmp_path / "t.duckdb"); bootstrap_schema(con)
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
    # No retries: messages.create called exactly once
    assert fake_client.messages.create.call_count == 1

@pytest.mark.asyncio
async def test_score_run_idempotency_duplicate_blocked(tmp_path):
    con = connect(tmp_path / "t.duckdb"); bootstrap_schema(con)
    qs = _seed_questions(con, 1)
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_fake_completion("B"))
    rid = start_run(con, model="m", prompt_version="v1", seed=1, subset_size=1, temperature=0.0)
    await score_run(con, fake_client, rid, "m", qs)
    # Re-score same questions for same run_id — should raise (UNIQUE violation)
    with pytest.raises(Exception, match=r"(?i)unique|duplicate|constraint"):
        await score_run(con, fake_client, rid, "m", qs)
```

- [ ] **Step 3: Run test — should fail**

Run: `uv run pytest tests/test_score.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 4: Create `eval_pipeline/score.py`**

```python
from __future__ import annotations
import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4
import duckdb
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from eval_pipeline.models import Question, ResponseRow
from eval_pipeline.prompts import SYSTEM_PROMPT, render_prompt, parse_letter
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
        api_error = "empty_completion" if not raw.strip() else None
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ResponseRow(
            response_id=uuid4(), run_id=run_id, question_id=q.question_id,
            raw_completion=raw, parsed_answer=parsed, is_correct=is_correct,
            latency_ms=latency_ms,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            api_error=api_error,
            responded_at=datetime.now(timezone.utc),
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ResponseRow(
            response_id=uuid4(), run_id=run_id, question_id=q.question_id,
            raw_completion="", parsed_answer=None, is_correct=None,
            latency_ms=latency_ms, input_tokens=0, output_tokens=0,
            api_error=type(e).__name__ + ": " + str(e)[:200],
            responded_at=datetime.now(timezone.utc),
        )

def _insert_responses(con: duckdb.DuckDBPyConnection, rows: list[ResponseRow]) -> None:
    for r in rows:
        con.execute(
            """INSERT INTO raw_eval_responses VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [str(r.response_id), str(r.run_id), r.question_id, r.raw_completion,
             r.parsed_answer, r.is_correct, r.latency_ms,
             r.input_tokens, r.output_tokens, r.api_error, r.responded_at],
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
        for i in range(0, len(questions), batch_size):
            chunk = questions[i : i + batch_size]
            results = await asyncio.gather(*[bounded(q) for q in chunk])
            _insert_responses(con, results)
        finish_run(con, run_id, status="completed")
    except Exception as e:
        mark_run_failed(con, run_id, f"{type(e).__name__}: {e}")
        raise
```

- [ ] **Step 5: Run tests — should pass**

Run: `uv run pytest tests/test_score.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add eval_pipeline/score.py tests/test_score.py tests/fixtures/anthropic_responses.json
git commit -m "feat: async scoring with retries, idempotency, honest empty-completion handling"
```

---

## Task 8: CLI (`cli.py`)

**Files:**
- Create: `eval_pipeline/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test `tests/test_cli.py`**

```python
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from typer.testing import CliRunner
from eval_pipeline.cli import app

runner = CliRunner()

def test_cli_load_creates_questions(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_DB_PATH", str(tmp_path / "t.duckdb"))
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = ([
            {"subject": "math", "question": "q", "choices": ["a","b","c","d"], "answer": 1}
        ], "sha")
        result = runner.invoke(app, ["load"])
    assert result.exit_code == 0, result.output
    assert "Loaded 1" in result.output

def test_cli_score_requires_model(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_DB_PATH", str(tmp_path / "t.duckdb"))
    result = runner.invoke(app, ["score"])  # missing --model
    assert result.exit_code != 0

@pytest.mark.integration
def test_cli_score_smoke_live_api(tmp_path, monkeypatch):
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    monkeypatch.setenv("EVAL_DB_PATH", str(tmp_path / "t.duckdb"))
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = ([
            {"subject": "math", "question": "What is 2+2?",
             "choices": ["3","4","5","6"], "answer": 1}
        ], "sha")
        runner.invoke(app, ["load"])
    result = runner.invoke(app, [
        "score", "--model", "claude-haiku-4-5-20251001",
        "--subset-size", "1", "--seed", "42",
    ])
    assert result.exit_code == 0, result.output
```

- [ ] **Step 2: Run test — should fail**

Run: `uv run pytest tests/test_cli.py -v -m "not integration"`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `eval_pipeline/cli.py`**

```python
from __future__ import annotations
import asyncio
import os
import random
from pathlib import Path
from typing import Optional
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
    "high_school_physics", "high_school_chemistry", "high_school_biology",
    "high_school_mathematics", "world_history", "philosophy",
]

def _db_path() -> Path:
    return Path(os.getenv("EVAL_DB_PATH", "data/eval.duckdb"))

@app.command()
def load(
    subjects: Optional[list[str]] = typer.Option(None, "--subject", help="Filter to subjects"),
) -> None:
    """Load MMLU into raw_mmlu_questions."""
    con = connect(_db_path())
    bootstrap_schema(con)
    n = load_mmlu_to_warehouse(con, subjects=subjects or DEFAULT_SUBJECTS)
    typer.echo(f"Loaded {n} questions")

@app.command()
def score(
    model: str = typer.Option(..., "--model", help="Model id, e.g. claude-haiku-4-5-20251001"),
    subset_size: int = typer.Option(200, "--subset-size"),
    seed: int = typer.Option(42, "--seed"),
    temperature: float = typer.Option(0.0, "--temperature"),
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
        raise typer.Exit("No questions loaded. Run `load` first.")
    rng.shuffle(rows)
    rows = rows[:subset_size]
    questions = [
        Question(
            question_id=r[0], subject=r[1], question=r[2],
            choice_a=r[3], choice_b=r[4], choice_c=r[5], choice_d=r[6],
            answer=r[7], dataset_version=r[8],
        ) for r in rows
    ]
    rid = start_run(con, model=model, prompt_version=PROMPT_TEMPLATE_VERSION,
                    seed=seed, subset_size=len(questions), temperature=temperature)
    client = AsyncAnthropic()
    asyncio.run(score_run(con, client, rid, model, questions))
    typer.echo(f"Run complete: {rid}")

@app.command()
def run(
    models: list[str] = typer.Option(
        ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
        "--model", help="Models to score (can repeat)",
    ),
    n_runs: int = typer.Option(3, "--n-runs"),
    subset_size: int = typer.Option(200, "--subset-size"),
    base_seed: int = typer.Option(42, "--seed"),
) -> None:
    """Run all models × n_runs."""
    for m in models:
        for i in range(n_runs):
            seed = base_seed + i
            typer.echo(f"--- {m} run {i+1}/{n_runs} (seed={seed}) ---")
            score(model=m, subset_size=subset_size, seed=seed, temperature=0.0)

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/test_cli.py -v -m "not integration"`
Expected: 2 passed, 1 skipped/deselected

- [ ] **Step 5: Verify entrypoint works**

Run: `uv run python -m eval_pipeline.cli --help`
Expected: usage text showing `load`, `score`, `run` commands.

- [ ] **Step 6: Commit**

```bash
git add eval_pipeline/cli.py tests/test_cli.py
git commit -m "feat: typer CLI with load/score/run commands"
```

---

## Task 9: dbt project scaffold

**Files:**
- Create: `dbt/dbt_project.yml`, `dbt/profiles.yml`, `dbt/models/.gitkeep`

- [ ] **Step 1: Create `dbt/dbt_project.yml`**

```yaml
name: 'eval_pipeline'
version: '0.1.0'
config-version: 2
profile: 'eval_pipeline'

model-paths: ["models"]
macro-paths: ["macros"]
seed-paths: ["seeds"]
test-paths: ["tests"]
target-path: "target"
clean-targets: ["target", "dbt_packages"]

models:
  eval_pipeline:
    staging:
      +materialized: view
    intermediate:
      +materialized: view
    marts:
      +materialized: table

seeds:
  eval_pipeline:
    +quote_columns: false
```

- [ ] **Step 2: Create `dbt/profiles.yml`**

```yaml
eval_pipeline:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "{{ env_var('EVAL_DB_PATH', '../data/eval.duckdb') }}"
      threads: 4
    ci:
      type: duckdb
      path: "{{ env_var('EVAL_DB_PATH', '../data/ci.duckdb') }}"
      threads: 2
    snowflake:
      # Portability target. Not required to run; documents that dbt can ship to a real warehouse.
      type: snowflake
      account: "{{ env_var('SNOWFLAKE_ACCOUNT', '') }}"
      user: "{{ env_var('SNOWFLAKE_USER', '') }}"
      password: "{{ env_var('SNOWFLAKE_PASSWORD', '') }}"
      role: "{{ env_var('SNOWFLAKE_ROLE', 'ANALYST') }}"
      database: "{{ env_var('SNOWFLAKE_DB', 'EVAL') }}"
      warehouse: "{{ env_var('SNOWFLAKE_WH', 'COMPUTE_WH') }}"
      schema: "{{ env_var('SNOWFLAKE_SCHEMA', 'MAIN') }}"
      threads: 4
```

- [ ] **Step 3: Verify dbt project loads**

Run: `cd dbt && uv run dbt debug --profiles-dir .`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add dbt/dbt_project.yml dbt/profiles.yml
git commit -m "feat: dbt project scaffold (duckdb dev + snowflake portable)"
```

---

## Task 10: dbt sources

**Files:**
- Create: `dbt/models/sources/sources.yml`

- [ ] **Step 1: Create `dbt/models/sources/sources.yml`**

```yaml
version: 2

sources:
  - name: raw
    description: "Immutable landing tables populated by eval_pipeline (Python)."
    schema: main
    tables:
      - name: raw_mmlu_questions
        description: "MMLU questions loaded from HuggingFace cais/mmlu."
        columns:
          - name: question_id
            description: "Stable 16-char hash of subject+question."
            data_tests: [unique, not_null]
          - name: answer
            data_tests:
              - not_null
              - accepted_values:
                  values: ['A', 'B', 'C', 'D']
      - name: raw_eval_runs
        description: "Run ledger. One row per `score` invocation."
        loaded_at_field: started_at
        freshness:
          warn_after: {count: 7, period: day}
          error_after: {count: 14, period: day}
        columns:
          - name: run_id
            data_tests: [unique, not_null]
          - name: status
            data_tests:
              - accepted_values:
                  values: ['completed', 'failed', 'partial']
      - name: raw_eval_responses
        description: "One row per (run, question) API call."
        columns:
          - name: response_id
            data_tests: [unique, not_null]
          - name: run_id
            data_tests:
              - not_null
              - relationships:
                  to: source('raw', 'raw_eval_runs')
                  field: run_id
          - name: question_id
            data_tests:
              - not_null
              - relationships:
                  to: source('raw', 'raw_mmlu_questions')
                  field: question_id
```

- [ ] **Step 2: Commit**

```bash
git add dbt/models/sources/sources.yml
git commit -m "feat: dbt source definitions with tests + freshness"
```

---

## Task 11: dbt staging models

**Files:**
- Create: `dbt/models/staging/stg_mmlu_questions.sql`, `stg_eval_runs.sql`, `stg_eval_responses.sql`, `staging.yml`

- [ ] **Step 1: Create `dbt/models/staging/stg_mmlu_questions.sql`**

```sql
with source as (
    select * from {{ source('raw', 'raw_mmlu_questions') }}
)

select
    question_id,
    lower(subject) as subject,
    question,
    choice_a,
    choice_b,
    choice_c,
    choice_d,
    answer,
    dataset_version,
    loaded_at
from source
```

- [ ] **Step 2: Create `dbt/models/staging/stg_eval_runs.sql`**

```sql
with source as (
    select * from {{ source('raw', 'raw_eval_runs') }}
)

select
    run_id,
    model,
    model_provider,
    prompt_version,
    seed,
    subset_size,
    temperature,
    started_at,
    finished_at,
    status,
    error_message,
    datediff('millisecond', started_at, finished_at) as duration_ms
from source
```

- [ ] **Step 3: Create `dbt/models/staging/stg_eval_responses.sql`**

```sql
with source as (
    select * from {{ source('raw', 'raw_eval_responses') }}
)

select
    response_id,
    run_id,
    question_id,
    raw_completion,
    parsed_answer,
    is_correct,
    coalesce(is_correct, false) as is_correct_strict,
    (parsed_answer is null) as is_unparseable,
    latency_ms,
    input_tokens,
    output_tokens,
    api_error,
    responded_at
from source
```

- [ ] **Step 4: Create `dbt/models/staging/staging.yml`**

```yaml
version: 2

models:
  - name: stg_mmlu_questions
    columns:
      - name: question_id
        data_tests: [unique, not_null]
      - name: answer
        data_tests:
          - accepted_values:
              values: ['A','B','C','D']
  - name: stg_eval_runs
    columns:
      - name: run_id
        data_tests: [unique, not_null]
      - name: status
        data_tests:
          - accepted_values:
              values: ['completed','failed','partial']
  - name: stg_eval_responses
    columns:
      - name: response_id
        data_tests: [unique, not_null]
      - name: run_id
        data_tests:
          - relationships:
              to: ref('stg_eval_runs')
              field: run_id
      - name: question_id
        data_tests:
          - relationships:
              to: ref('stg_mmlu_questions')
              field: question_id
```

- [ ] **Step 5: Smoke build (requires a populated DuckDB from Task 4/8)**

Run from repo root: `uv run python -m eval_pipeline.cli load --subject high_school_physics`
Then: `cd dbt && uv run dbt build --profiles-dir . --select staging`
Expected: 3 models built, no failures. (Tests on relationships may fail if no responses yet — that's fine; we'll re-test after Task 18.)

- [ ] **Step 6: Commit**

```bash
git add dbt/models/staging/
git commit -m "feat: dbt staging models with type casts and is_unparseable flag"
```

---

## Task 12: MMLU subject groups seed

**Files:**
- Create: `dbt/seeds/mmlu_subject_groups.csv`

- [ ] **Step 1: Create `dbt/seeds/mmlu_subject_groups.csv`**

```csv
subject,subject_group
abstract_algebra,stem
anatomy,stem
astronomy,stem
business_ethics,social_science
clinical_knowledge,stem
college_biology,stem
college_chemistry,stem
college_computer_science,stem
college_mathematics,stem
college_medicine,stem
college_physics,stem
computer_security,stem
conceptual_physics,stem
econometrics,social_science
electrical_engineering,stem
elementary_mathematics,stem
formal_logic,humanities
global_facts,humanities
high_school_biology,stem
high_school_chemistry,stem
high_school_computer_science,stem
high_school_european_history,humanities
high_school_geography,social_science
high_school_government_and_politics,social_science
high_school_macroeconomics,social_science
high_school_mathematics,stem
high_school_microeconomics,social_science
high_school_physics,stem
high_school_psychology,social_science
high_school_statistics,stem
high_school_us_history,humanities
high_school_world_history,humanities
human_aging,stem
human_sexuality,social_science
international_law,social_science
jurisprudence,social_science
logical_fallacies,humanities
machine_learning,stem
management,social_science
marketing,social_science
medical_genetics,stem
miscellaneous,other
moral_disputes,humanities
moral_scenarios,humanities
nutrition,stem
philosophy,humanities
prehistory,humanities
professional_accounting,social_science
professional_law,social_science
professional_medicine,stem
professional_psychology,social_science
public_relations,social_science
security_studies,social_science
sociology,social_science
us_foreign_policy,social_science
virology,stem
world_religions,humanities
```

- [ ] **Step 2: Build seed**

Run: `cd dbt && uv run dbt seed --profiles-dir .`
Expected: `1 of 1 OK`

- [ ] **Step 3: Commit**

```bash
git add dbt/seeds/mmlu_subject_groups.csv
git commit -m "feat: seed mapping 57 MMLU subjects to 4 groups"
```

---

## Task 13: dbt intermediate models

**Files:**
- Create: `dbt/models/intermediate/int_scored_responses.sql`, `int_run_summary.sql`, `intermediate.yml`

- [ ] **Step 1: Create `dbt/models/intermediate/int_scored_responses.sql`**

```sql
with responses as (
    select * from {{ ref('stg_eval_responses') }}
),
runs as (
    select * from {{ ref('stg_eval_runs') }}
),
questions as (
    select * from {{ ref('stg_mmlu_questions') }}
),
groups as (
    select * from {{ ref('mmlu_subject_groups') }}
)

select
    r.response_id,
    r.run_id,
    r.question_id,
    ru.model,
    ru.prompt_version,
    ru.seed,
    ru.started_at as run_started_at,
    q.subject,
    coalesce(g.subject_group, 'other') as subject_group,
    q.answer as ground_truth,
    r.parsed_answer,
    r.is_correct,
    r.is_correct_strict,
    r.is_unparseable,
    r.latency_ms,
    r.input_tokens,
    r.output_tokens,
    r.api_error
from responses r
join runs ru using (run_id)
join questions q using (question_id)
left join groups g using (subject)
where ru.status = 'completed'
```

- [ ] **Step 2: Create `dbt/models/intermediate/int_run_summary.sql`**

```sql
with scored as (
    select * from {{ ref('int_scored_responses') }}
)

select
    run_id,
    model,
    prompt_version,
    run_started_at,
    count(*) as n_responses,
    sum(case when is_unparseable then 0 else 1 end) as n_parsed,
    sum(case when is_correct_strict then 1 else 0 end) as n_correct,
    avg(case when is_correct_strict then 1.0 else 0.0 end) as pass_rate,
    avg(case when is_unparseable then 1.0 else 0.0 end) as unparseable_rate,
    avg(latency_ms) as mean_latency_ms,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens
from scored
group by 1, 2, 3, 4
```

- [ ] **Step 3: Create `dbt/models/intermediate/intermediate.yml`**

```yaml
version: 2

models:
  - name: int_scored_responses
    description: "Responses joined to runs + questions + subject groups. Only completed runs."
    columns:
      - name: response_id
        data_tests: [unique, not_null]
      - name: subject_group
        data_tests:
          - accepted_values:
              values: ['stem','humanities','social_science','other']
  - name: int_run_summary
    description: "Per-run rollup: pass rate, unparseable rate, latency, tokens."
    columns:
      - name: run_id
        data_tests: [unique, not_null]
      - name: pass_rate
        data_tests:
          - dbt_utils.expression_is_true:
              expression: ">= 0 and pass_rate <= 1"
```

(Note: the `dbt_utils.expression_is_true` test requires `dbt-utils`; if not installed, replace with a singular test added in Task 17.)

- [ ] **Step 4: Commit**

```bash
git add dbt/models/intermediate/
git commit -m "feat: dbt intermediate models — scored responses + run summary"
```

---

## Task 14: Wilson CI macro

**Files:**
- Create: `dbt/macros/wilson_ci.sql`

- [ ] **Step 1: Create `dbt/macros/wilson_ci.sql`**

```sql
{# Wilson 95% confidence interval for a binomial proportion.

   Usage: {{ wilson_ci(successes_col='n_correct', trials_col='n_responses') }}
   Returns two columns: ci_lower, ci_upper.
#}

{% macro wilson_ci(successes_col, trials_col, z=1.96) -%}
    case when {{ trials_col }} = 0 then null else
        (
            ({{ successes_col }}::double / {{ trials_col }})
            + ({{ z }} * {{ z }}) / (2.0 * {{ trials_col }})
            - {{ z }} * sqrt(
                ({{ successes_col }}::double / {{ trials_col }})
                * (1 - {{ successes_col }}::double / {{ trials_col }})
                / {{ trials_col }}
                + ({{ z }} * {{ z }}) / (4.0 * {{ trials_col }} * {{ trials_col }})
            )
        ) / (1 + ({{ z }} * {{ z }}) / {{ trials_col }})
    end as ci_lower,
    case when {{ trials_col }} = 0 then null else
        (
            ({{ successes_col }}::double / {{ trials_col }})
            + ({{ z }} * {{ z }}) / (2.0 * {{ trials_col }})
            + {{ z }} * sqrt(
                ({{ successes_col }}::double / {{ trials_col }})
                * (1 - {{ successes_col }}::double / {{ trials_col }})
                / {{ trials_col }}
                + ({{ z }} * {{ z }}) / (4.0 * {{ trials_col }} * {{ trials_col }})
            )
        ) / (1 + ({{ z }} * {{ z }}) / {{ trials_col }})
    end as ci_upper
{%- endmacro %}
```

- [ ] **Step 2: Commit**

```bash
git add dbt/macros/wilson_ci.sql
git commit -m "feat: Wilson 95% confidence interval macro"
```

---

## Task 15: Mart — pass rate by model

**Files:**
- Create: `dbt/models/marts/mart_pass_rate_by_model.sql`

- [ ] **Step 1: Create `dbt/models/marts/mart_pass_rate_by_model.sql`**

```sql
with scored as (
    select * from {{ ref('int_scored_responses') }}
),
agg as (
    select
        model,
        subject_group,
        count(*) as n_responses,
        sum(case when is_correct_strict then 1 else 0 end) as n_correct,
        count(distinct run_id) as n_runs
    from scored
    group by 1, 2
)

select
    model,
    subject_group,
    n_responses,
    n_runs,
    n_correct::double / nullif(n_responses, 0) as pass_rate,
    {{ wilson_ci('n_correct', 'n_responses') }}
from agg
```

- [ ] **Step 2: Commit**

```bash
git add dbt/models/marts/mart_pass_rate_by_model.sql
git commit -m "feat: mart_pass_rate_by_model with Wilson CIs"
```

---

## Task 16: Marts — category breakdown + run drift

**Files:**
- Create: `dbt/models/marts/mart_category_breakdown.sql`, `mart_run_drift.sql`, `marts.yml`

- [ ] **Step 1: Create `dbt/models/marts/mart_category_breakdown.sql`**

```sql
with scored as (
    select * from {{ ref('int_scored_responses') }}
)

select
    model,
    subject,
    count(*) as n_responses,
    count(distinct run_id) as n_runs,
    sum(case when is_correct_strict then 1 else 0 end)::double
        / nullif(count(*), 0) as pass_rate,
    avg(case when is_unparseable then 1.0 else 0.0 end) as unparseable_rate
from scored
group by 1, 2
```

- [ ] **Step 2: Create `dbt/models/marts/mart_run_drift.sql`**

```sql
with summary as (
    select * from {{ ref('int_run_summary') }}
),
ordered as (
    select
        model,
        run_id,
        run_started_at,
        pass_rate,
        n_responses,
        row_number() over (partition by model order by run_started_at) as run_seq
    from summary
),
first_run as (
    select model, pass_rate as first_pass_rate
    from ordered
    where run_seq = 1
)

select
    o.model,
    o.run_id,
    o.run_seq,
    o.run_started_at,
    o.pass_rate,
    o.n_responses,
    o.pass_rate - fr.first_pass_rate as delta_from_first_run
from ordered o
left join first_run fr using (model)
order by o.model, o.run_seq
```

- [ ] **Step 3: Create `dbt/models/marts/marts.yml`**

```yaml
version: 2

models:
  - name: mart_pass_rate_by_model
    description: "Headline mart: per-model, per-subject-group pass rate with Wilson 95% CI."
    columns:
      - name: model
        data_tests: [not_null]
      - name: pass_rate
        data_tests: [not_null]
  - name: mart_category_breakdown
    description: "Per-model, per-subject pass rate. Finer-grained than subject_group."
  - name: mart_run_drift
    description: "Per-model, per-run pass rate ordered by start time; drift legible via delta_from_first_run."
    columns:
      - name: run_seq
        data_tests: [not_null]
```

- [ ] **Step 4: Commit**

```bash
git add dbt/models/marts/
git commit -m "feat: marts for category breakdown and run drift"
```

---

## Task 17: dbt singular tests

**Files:**
- Create: `dbt/tests/assert_no_orphan_responses.sql`, `assert_pass_rate_in_range.sql`, `assert_run_seq_dense.sql`

- [ ] **Step 1: Create `dbt/tests/assert_no_orphan_responses.sql`**

```sql
-- Every raw_eval_responses.run_id must exist in raw_eval_runs.
select r.run_id
from {{ source('raw', 'raw_eval_responses') }} r
left join {{ source('raw', 'raw_eval_runs') }} ru using (run_id)
where ru.run_id is null
```

- [ ] **Step 2: Create `dbt/tests/assert_pass_rate_in_range.sql`**

```sql
-- pass_rate in marts must be between 0 and 1.
select 'mart_pass_rate_by_model' as src, pass_rate
from {{ ref('mart_pass_rate_by_model') }}
where pass_rate < 0 or pass_rate > 1
union all
select 'mart_category_breakdown', pass_rate
from {{ ref('mart_category_breakdown') }}
where pass_rate < 0 or pass_rate > 1
union all
select 'mart_run_drift', pass_rate
from {{ ref('mart_run_drift') }}
where pass_rate < 0 or pass_rate > 1
```

- [ ] **Step 3: Create `dbt/tests/assert_run_seq_dense.sql`**

```sql
-- run_seq must be dense (1, 2, 3, ...) per model — no gaps.
with seq as (
    select model, run_seq,
           lag(run_seq) over (partition by model order by run_seq) as prev_seq
    from {{ ref('mart_run_drift') }}
)
select model, run_seq, prev_seq
from seq
where prev_seq is not null and run_seq <> prev_seq + 1
```

- [ ] **Step 4: Build entire project (requires populated DB)**

Run: `cd dbt && uv run dbt build --profiles-dir .`
Expected: all models + tests pass. (May skip drift tests until 2+ runs exist; that's fine.)

- [ ] **Step 5: Commit**

```bash
git add dbt/tests/
git commit -m "feat: dbt singular tests for orphan responses, pass-rate range, dense run_seq"
```

---

## Task 18: Streamlit dashboard

**Files:**
- Create: `dashboard/app.py`

- [ ] **Step 1: Create `dashboard/app.py`**

```python
from __future__ import annotations
import os
from pathlib import Path
import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LLM Eval Pipeline", layout="wide")
st.title("LLM Eval Pipeline — MMLU vs Claude")

DB_PATH = Path(os.getenv("EVAL_DB_PATH", "data/eval.duckdb"))

@st.cache_data(ttl=60)
def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()

if not DB_PATH.exists():
    st.error(f"No DuckDB at {DB_PATH}. Run `make all` or `python -m eval_pipeline.cli run`.")
    st.stop()

st.header("Pass rate by model (with 95% Wilson CIs)")
df_pass = query("SELECT * FROM mart_pass_rate_by_model ORDER BY model, subject_group")
if df_pass.empty:
    st.warning("No data in mart_pass_rate_by_model. Run scoring + dbt build first.")
else:
    pivot = df_pass.pivot(index="subject_group", columns="model", values="pass_rate")
    st.bar_chart(pivot)
    with st.expander("Raw mart_pass_rate_by_model"):
        st.dataframe(df_pass, use_container_width=True)

st.header("Category breakdown (top + bottom 10 categories per model)")
df_cat = query("SELECT * FROM mart_category_breakdown ORDER BY model, pass_rate")
if not df_cat.empty:
    models = sorted(df_cat["model"].unique())
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        sub = df_cat[df_cat["model"] == m]
        col.subheader(m)
        col.dataframe(
            pd.concat([sub.head(5), sub.tail(5)])[["subject", "pass_rate", "unparseable_rate"]],
            use_container_width=True,
        )

st.header("Drift across runs")
df_drift = query("SELECT * FROM mart_run_drift ORDER BY model, run_seq")
if not df_drift.empty:
    chart = df_drift.pivot(index="run_seq", columns="model", values="pass_rate")
    st.line_chart(chart)
    with st.expander("delta_from_first_run"):
        st.dataframe(
            df_drift[["model", "run_seq", "pass_rate", "delta_from_first_run"]],
            use_container_width=True,
        )

st.caption(f"Source: {DB_PATH} • Refreshed every 60s")
```

- [ ] **Step 2: Smoke-test dashboard**

Run: `uv run streamlit run dashboard/app.py`
Expected: browser opens to localhost:8501; either charts render (if data present) or warning shown.
Stop with Ctrl-C.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: Streamlit dashboard reading marts from DuckDB"
```

---

## Task 19: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --frozen
      - name: Lint
        run: uv run ruff check .
      - name: Type check
        run: uv run mypy eval_pipeline
      - name: Unit tests
        run: uv run pytest -m "not integration" -v

  dbt:
    runs-on: ubuntu-latest
    needs: python
    env:
      EVAL_DB_PATH: ${{ github.workspace }}/data/ci.duckdb
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --frozen
      - name: Seed CI database from fixtures
        run: uv run python scripts/seed_ci_db.py
      - name: dbt seed + build + test
        working-directory: dbt
        run: |
          uv run dbt seed --profiles-dir . --target ci
          uv run dbt build --profiles-dir . --target ci --exclude tag:freshness
      - name: dbt docs
        working-directory: dbt
        run: uv run dbt docs generate --profiles-dir . --target ci
      - uses: actions/upload-artifact@v4
        with:
          name: dbt-docs
          path: dbt/target/
```

- [ ] **Step 2: Create `scripts/seed_ci_db.py`**

```python
"""Populate a CI DuckDB from test fixtures so dbt build can run without API keys."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from eval_pipeline.load import _hash_question_id
from eval_pipeline.warehouse import bootstrap_schema, connect

FIXTURE_Q = Path("tests/fixtures/mmlu_sample.json")
FIXTURE_R = Path("tests/fixtures/anthropic_responses.json")

def main() -> None:
    db = Path(os.getenv("EVAL_DB_PATH", "data/ci.duckdb"))
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    con = connect(db)
    bootstrap_schema(con)

    qs = json.loads(FIXTURE_Q.read_text())
    letters = ["A", "B", "C", "D"]
    qids = []
    for q in qs:
        qid = _hash_question_id(q["subject"], q["question"])
        qids.append(qid)
        con.execute(
            """INSERT INTO raw_mmlu_questions VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [qid, q["subject"], q["question"],
             q["choices"][0], q["choices"][1], q["choices"][2], q["choices"][3],
             letters[q["answer"]], "ci-fixture", datetime.now(timezone.utc)],
        )

    # Seed 2 runs per model so mart_run_drift has rows
    responses = json.loads(FIXTURE_R.read_text())
    for model in ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]:
        for run_idx in range(2):
            rid = str(uuid4())
            con.execute(
                """INSERT INTO raw_eval_runs
                   VALUES (?, ?, 'anthropic', 'v1@ci', ?, ?, 0.0, ?, ?, 'completed', NULL)""",
                [rid, model, 42 + run_idx, len(qids),
                 datetime.now(timezone.utc), datetime.now(timezone.utc)],
            )
            for i, qid in enumerate(qids):
                r = responses[i % len(responses)]
                from eval_pipeline.prompts import parse_letter
                parsed = parse_letter(r["completion"])
                # ground truth for fixture questions, in order
                gt = [letters[q["answer"]] for q in qs][i]
                is_correct = (parsed == gt) if parsed else None
                api_err = "empty_completion" if not r["completion"].strip() else None
                con.execute(
                    """INSERT INTO raw_eval_responses
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    [str(uuid4()), rid, qid, r["completion"], parsed, is_correct,
                     100, r["input_tokens"], r["output_tokens"], api_err,
                     datetime.now(timezone.utc)],
                )
    print(f"Seeded CI db at {db}: {len(qids)} questions, 6 runs")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Tag freshness source-tests so CI excludes them**

Edit `dbt/models/sources/sources.yml` — find the `raw_eval_runs` block and add a tag inside `freshness`:

```yaml
      - name: raw_eval_runs
        loaded_at_field: started_at
        config:
          tags: [freshness]
        freshness:
          warn_after: {count: 7, period: day}
          error_after: {count: 14, period: day}
```

(Freshness sources need recent data — CI fixtures use now() so they pass anyway, but the tag is the standard escape hatch.)

- [ ] **Step 4: Smoke-test CI script locally**

Run from repo root: `EVAL_DB_PATH=data/ci.duckdb uv run python scripts/seed_ci_db.py`
Then: `cd dbt && EVAL_DB_PATH=../data/ci.duckdb uv run dbt build --profiles-dir . --target ci`
Expected: all models + tests pass.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml scripts/seed_ci_db.py dbt/models/sources/sources.yml
git commit -m "ci: GitHub Actions running ruff/mypy/pytest + dbt build on fixture data"
```

---

## Task 20: ADRs (`docs/decisions.md`)

**Files:**
- Create: `docs/decisions.md`

- [ ] **Step 1: Create `docs/decisions.md`**

```markdown
# Architecture Decision Records

## ADR-001: DuckDB as primary warehouse, Snowflake as portable target

**Status:** Accepted (2026-05-27)

**Context.** This is a portfolio project. It needs to be cloneable, runnable, and verifiable by any reviewer indefinitely. Snowflake's 30-day trial means the demo dies after a month. DuckDB is free forever, embedded, and zero-setup.

**Decision.** Use DuckDB as the default profile. Include a Snowflake profile in `profiles.yml` to demonstrate the dbt project is warehouse-portable. The Snowflake target is not in the critical path.

**Consequences.** Anyone can clone this repo, run `make all`, and see the dashboard in <10 minutes. The "I can ship this to a real warehouse" narrative is preserved in code, not just words.

---

## ADR-002: 3 Anthropic models × 3 runs each

**Status:** Accepted (2026-05-27)

**Context.** "Drift over time" requires multiple eval runs. Cost and time bound how many we can do. Wilson 95% CIs need n >= 2 to be meaningful.

**Decision.** Score 3 Anthropic models (Haiku 4.5, Sonnet 4.6, Opus 4.7) with 3 runs each at different sampling seeds. Total: 9 runs.

**Consequences.** Total API cost under $20 for a 200-question subset. Wilson CIs make the noise floor visible. Cross-model comparison is the dominant story; temporal drift is intentionally weak — flagged in README Limitations.

---

## ADR-003: Parsing lives in raw, not staging

**Status:** Accepted (2026-05-27)

**Context.** LLM completions are not always parseable into A/B/C/D. The parser is a regex stack. Bugs in the parser are likely; the cost of being wrong is having to re-call the API at $0.001-$0.015 per call.

**Decision.** Store `parsed_answer` and `is_correct` in `raw_eval_responses` alongside `raw_completion`. The raw completion is retained.

**Consequences.** "Re-parse historical runs with a better regex" is a SQL-only refresh — never an API rerun. The cost of parser experimentation is near-zero. Staging stays a pure projection.

---

## ADR-004: No judge-model scoring

**Status:** Accepted (2026-05-27)

**Context.** A more rigorous parser would use an LLM-as-judge to extract the answer from free-form completions. This handles refusals, equivocation, and "I think it's A but B is also reasonable" outputs.

**Decision.** Out of scope for this portfolio project. Use regex parsing. Track `unparseable_rate` as a first-class metric in `int_run_summary` so the regex's failures are visible.

**Consequences.** Some completions will be marked unparseable that a judge model could resolve. The Limitations section in the README calls this out explicitly. The shape of the pipeline supports adding a judge-model step later as a new column in `raw_eval_responses` — no schema rework needed.
```

- [ ] **Step 2: Commit**

```bash
git add docs/decisions.md
git commit -m "docs: 4 ADRs (warehouse choice, model setup, parsing boundary, no judge model)"
```

---

## Task 21: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# LLM Eval Pipeline

A reproducible analytics-engineering pipeline for LLM evaluation data — MMLU benchmark scored against three Claude models, modeled in dbt, served as a local Streamlit dashboard. Built to demonstrate the kind of plumbing an eval team needs from analytics engineering.

> ⚠️ Portfolio project. Three runs per model is not enough for real eval conclusions — see [Limitations](#limitations).

![dashboard screenshot placeholder](docs/dashboard.png)

## Why eval-data plumbing matters for an AI lab

Every public benchmark number you see — MMLU 87%, HumanEval 92% — sits on top of a pipeline that loaded a dataset, sampled it, ran a model against it, parsed completions, joined to ground truth, and aggregated. That pipeline rarely shows up in papers. When it breaks subtly — a prompt-template typo, a parser regex that drops 3% of completions as "unparseable", a dataset-version drift — the benchmark number lies and a research decision gets made on a lie.

Analytics engineering for an eval team means owning that pipeline like it's a financial reporting system: versioned schemas, lineage, freshness checks, separated parsing from scoring, reproducible reruns. This repo is what that looks like for a small MMLU subset.

## What this pipeline does

```
HuggingFace MMLU ──(load.py)──▶ raw_mmlu_questions ─┐
                                                     ├─▶ stg_* ─▶ int_* ─▶ mart_*
Anthropic API ───(score.py)──▶ raw_eval_responses ─┤
                              raw_eval_runs ────────┘
```

- **3 Claude models** (Haiku 4.5, Sonnet 4.6, Opus 4.7), **3 runs each**, scored against an MMLU subset.
- **dbt project** with `staging → intermediate → marts`. Three marts: `mart_pass_rate_by_model` (with Wilson 95% CIs), `mart_category_breakdown` (per-subject), `mart_run_drift`.
- **Streamlit dashboard** for exploration.
- **dbt docs lineage** at `docs/lineage.svg`.

## What's interesting in the code

- **Parsing is separated from scoring.** `parsed_answer` lives in `raw_eval_responses` alongside `raw_completion`. A bad regex doesn't require re-calling the API — it's a pure SQL refresh. → [`eval_pipeline/score.py`](eval_pipeline/score.py)
- **`unparseable_rate` is a first-class metric.** Honest evals distinguish "model got it wrong" from "parser couldn't tell." → [`dbt/models/intermediate/int_run_summary.sql`](dbt/models/intermediate/int_run_summary.sql)
- **Wilson 95% confidence intervals on pass rates.** Three runs per model isn't a lot — the CI math makes the noise floor visible. → [`dbt/macros/wilson_ci.sql`](dbt/macros/wilson_ci.sql)
- **Run ledger + idempotent inserts.** Every run gets a UUID, a captured prompt-template version + git SHA, and a UNIQUE constraint on `(run_id, question_id)`. Drift is legible because reruns can't silently merge. → [`eval_pipeline/runs.py`](eval_pipeline/runs.py)

## Running it yourself

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), an Anthropic API key.

```bash
git clone <this-repo>
cd llm-eval-pipeline
cp .env.example .env  # then edit ANTHROPIC_API_KEY
uv sync

# 1. Load MMLU into DuckDB (one-time, ~30s)
uv run python -m eval_pipeline.cli load

# 2. Score all 3 models × 3 runs (~5 min, ~$5-$20 in API)
uv run python -m eval_pipeline.cli run

# 3. Build the dbt project
cd dbt && uv run dbt seed --profiles-dir . && uv run dbt build --profiles-dir .

# 4. Launch the dashboard
cd .. && uv run streamlit run dashboard/app.py
```

Or: `make all`.

## Decisions

See [`docs/decisions.md`](docs/decisions.md) for 4 ADRs covering warehouse choice, model setup, parsing boundary, and judge-model scope.

## Limitations

Earnestness as signal:

- **Three runs per model is barely enough for the Wilson CI to be meaningful.** A real eval team runs hundreds. The CI bands in the dashboard are honest about this.
- **MMLU is a saturated benchmark.** It's used here as a known quantity for tooling, not as a state-of-the-art signal. Pass rates in the 80-90% range are expected; differences between models on a 200-question subset may be inside the CI.
- **The parser is regex-based.** A judge-model parser would handle long-form refusals and equivocation better. `unparseable_rate` is tracked so the regex's failures are visible.
- **Drift is shown across runs hours apart, not weeks.** Real drift detection wants cron history.

## License

MIT.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with hook, differentiators, and earnest limitations"
```

---

## Task 22: Makefile + lineage export + final polish

**Files:**
- Create: `Makefile`, `docs/lineage.svg`

- [ ] **Step 1: Create `Makefile`**

```makefile
.PHONY: all install load score build dashboard test lint typecheck clean

EVAL_DB_PATH ?= data/eval.duckdb
export EVAL_DB_PATH

all: install load score build
	@echo "Done. Run 'make dashboard' to view."

install:
	uv sync

load:
	uv run python -m eval_pipeline.cli load

score:
	uv run python -m eval_pipeline.cli run

build:
	cd dbt && uv run dbt seed --profiles-dir . && uv run dbt build --profiles-dir .

dashboard:
	uv run streamlit run dashboard/app.py

test:
	uv run pytest -m "not integration" -v

lint:
	uv run ruff check .

typecheck:
	uv run mypy eval_pipeline

docs:
	cd dbt && uv run dbt docs generate --profiles-dir .
	@echo "dbt docs in dbt/target/. Open dbt/target/index.html in a browser."

clean:
	rm -rf data/*.duckdb data/*.duckdb.wal dbt/target dbt/dbt_packages
```

- [ ] **Step 2: Generate dbt lineage**

Run: `make build && make docs`
Then open `dbt/target/index.html` in a browser, navigate to the lineage graph, screenshot it, save as `docs/lineage.svg` (or PNG — README link works either way).

- [ ] **Step 3: Run the full test + lint suite one more time**

Run:
```
uv run ruff check .
uv run mypy eval_pipeline
uv run pytest -m "not integration" -v
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add Makefile docs/lineage.svg
git commit -m "chore: Makefile + lineage diagram for README"
```

- [ ] **Step 5: Final smoke test — full pipeline**

Run: `make clean && make all && make dashboard`
Expected: dashboard renders all three sections with real data; no errors.

- [ ] **Step 6: Push to GitHub**

```bash
gh repo create llm-eval-pipeline --public --source=. --remote=origin --push
```

(Or skip if you want to keep it private; the local repo is complete either way.)

---

## Self-Review Checklist

**Spec coverage** — every section of the spec maps to one or more tasks:
- §2 Stack → Task 1 (pyproject), Task 9 (dbt profiles), Task 18 (Streamlit), Task 19 (CI)
- §3 Repo layout → Task 1 + every task creating its listed file
- §4 Data flow → emerges from Tasks 4, 7, 11-17, 18
- §5 Raw schema → Task 3
- §6 dbt layer → Tasks 10-17
- §7 Scoring logic + failure modes + idempotency + CLI → Tasks 5, 6, 7, 8
- §8 Testing → Tasks 2-8 (TDD inline) + Task 17 (dbt singular tests)
- §9 CI → Task 19
- §10 README + ADRs → Tasks 20, 21
- §11 Out of scope → no tasks (correctly)
- §12 Open questions → none
- §13 Effort estimate → matches task ordering

**No placeholders.** All code blocks are complete (zero "TODO", "TBD", or "fill in" instructions).

**Type consistency.**
- `Question`, `ResponseRow`, `RunMeta` defined in Task 2; used consistently in Tasks 4, 6, 7, 8.
- `_hash_question_id` defined Task 4; reused in Task 7 test setup and Task 19 CI seed.
- `parse_letter`, `render_prompt`, `SYSTEM_PROMPT`, `PROMPT_TEMPLATE_VERSION` defined Task 5; used in Tasks 7, 8, 19.
- `start_run`, `finish_run`, `mark_run_failed`, `get_run` defined Task 6; used in Tasks 7, 8.
- `score_run` signature: `(con, client, run_id, model, questions, batch_size=10)` consistent across Tasks 7, 8.
- dbt model names (`stg_*`, `int_*`, `mart_*`) referenced consistently across Tasks 11-18.

No issues found.
