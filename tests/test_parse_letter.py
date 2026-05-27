import pytest

from eval_pipeline.models import Question
from eval_pipeline.prompts import (
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
    parse_letter,
    render_prompt,
)


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
