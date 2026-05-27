from datetime import UTC, datetime
from uuid import uuid4

import pytest

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
        api_error=None, responded_at=datetime.now(UTC),
    )
    assert r.parsed_answer is None
    assert r.is_correct is None

def test_run_meta_status_enum():
    with pytest.raises(ValueError):
        RunMeta(
            run_id=uuid4(), model="claude-haiku-4-5", model_provider="anthropic",
            prompt_version="sha", seed=42, subset_size=10, temperature=0.0,
            started_at=datetime.now(UTC), finished_at=None,
            status="bogus", error_message=None,
        )
