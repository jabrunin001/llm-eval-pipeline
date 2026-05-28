import os
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from eval_pipeline.cli import app

runner = CliRunner()


def test_cli_load_creates_questions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DB_PATH", str(tmp_path / "t.duckdb"))
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = ([
            {"subject": "math", "question": "q", "choices": ["a", "b", "c", "d"], "answer": 1}
        ], "sha")
        result = runner.invoke(app, ["load"])
    assert result.exit_code == 0, result.output
    assert "Loaded 1" in result.output


def test_cli_score_requires_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DB_PATH", str(tmp_path / "t.duckdb"))
    result = runner.invoke(app, ["score"])  # missing --model
    assert result.exit_code != 0


@pytest.mark.integration
def test_cli_score_smoke_live_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    monkeypatch.setenv("EVAL_DB_PATH", str(tmp_path / "t.duckdb"))
    with patch("eval_pipeline.load._fetch_mmlu") as m:
        m.return_value = ([
            {"subject": "math", "question": "What is 2+2?",
             "choices": ["3", "4", "5", "6"], "answer": 1}
        ], "sha")
        runner.invoke(app, ["load"])
    result = runner.invoke(app, [
        "score", "--model", "claude-haiku-4-5-20251001",
        "--subset-size", "1", "--seed", "42",
    ])
    assert result.exit_code == 0, result.output
