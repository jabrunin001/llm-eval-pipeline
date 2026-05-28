"""Populate a CI DuckDB from test fixtures so dbt build can run without API keys."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from eval_pipeline.load import _hash_question_id
from eval_pipeline.prompts import parse_letter
from eval_pipeline.warehouse import bootstrap_schema, connect

FIXTURE_Q = Path("tests/fixtures/mmlu_sample.json")
FIXTURE_R = Path("tests/fixtures/anthropic_responses.json")
LETTERS = ["A", "B", "C", "D"]


def main() -> None:
    db = Path(os.getenv("EVAL_DB_PATH", "data/ci.duckdb"))
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    con = connect(db)
    bootstrap_schema(con)

    qs = json.loads(FIXTURE_Q.read_text())
    qids: list[str] = []
    for q in qs:
        qid = _hash_question_id(q["subject"], q["question"])
        qids.append(qid)
        con.execute(
            "INSERT INTO raw_mmlu_questions VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                qid, q["subject"], q["question"],
                q["choices"][0], q["choices"][1], q["choices"][2], q["choices"][3],
                LETTERS[q["answer"]], "ci-fixture", datetime.now(UTC),
            ],
        )

    responses = json.loads(FIXTURE_R.read_text())
    for model in ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]:
        for run_idx in range(2):
            rid = str(uuid4())
            con.execute(
                "INSERT INTO raw_eval_runs VALUES "
                "(?, ?, 'anthropic', 'v1@ci', ?, ?, 0.0, ?, ?, 'completed', NULL)",
                [
                    rid, model, 42 + run_idx, len(qids),
                    datetime.now(UTC), datetime.now(UTC),
                ],
            )
            for i, qid in enumerate(qids):
                r = responses[i % len(responses)]
                parsed = parse_letter(r["completion"])
                gt = LETTERS[qs[i]["answer"]]
                is_correct = (parsed == gt) if parsed else None
                api_err = (
                    "empty_completion" if not r["completion"].strip()
                    else ("unparseable" if parsed is None else None)
                )
                con.execute(
                    "INSERT INTO raw_eval_responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        str(uuid4()), rid, qid, r["completion"], parsed, is_correct,
                        100, r["input_tokens"], r["output_tokens"], api_err,
                        datetime.now(UTC),
                    ],
                )
    print(f"Seeded CI db at {db}: {len(qids)} questions, 6 runs")


if __name__ == "__main__":
    main()
