from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import duckdb

_LETTERS = ["A", "B", "C", "D"]


def _hash_question_id(subject: str, question: str) -> str:
    return hashlib.sha256(f"{subject}|{question}".encode()).hexdigest()[:16]


def _fetch_mmlu(subjects: list[str] | None) -> tuple[list[dict[str, Any]], str]:
    """Fetch MMLU from HuggingFace. Returns (rows, dataset_version_sha).

    Lazy-imports `datasets` so unit tests that patch this function don't need it.
    Mocked in tests.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("cais/mmlu", "all", split="test")
        rows: list[dict[str, Any]] = []
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
    except Exception as e:
        raise RuntimeError(f"Failed to fetch MMLU from HuggingFace: {e}") from e


def load_mmlu_to_warehouse(
    con: duckdb.DuckDBPyConnection,
    subjects: list[str] | None = None,
) -> int:
    rows, version = _fetch_mmlu(subjects)
    now = datetime.now(UTC)
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
