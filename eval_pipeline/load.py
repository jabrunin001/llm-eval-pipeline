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
