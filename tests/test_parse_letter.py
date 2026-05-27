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


def test_render_prompt_preserves_braces_in_data():
    q = Question(
        question_id="x", subject="math", question="Solve {x+1}=2",
        choice_a="x={1}", choice_b="x=0", choice_c="x={2}", choice_d="x={3}",
        answer="A", dataset_version="sha",
    )
    out = render_prompt(q)
    assert "{x+1}=2" in out
    assert "x={1}" in out


@pytest.mark.parametrize("raw,expected", [
    ("A or B", None),               # ambiguous
    ("The answer is B or C", "B"),  # "answer is" targets first
    ("Options A, B, C, and D", None),  # all four
    ("Just A here, no others", "A"),  # only one letter present
])
def test_parse_letter_ambiguity(raw, expected):
    assert parse_letter(raw) == expected


def test_parse_letter_avoids_letter_hyphen_words():
    assert parse_letter("D-level performance") is None
    assert parse_letter("B-cell receptors are...") is None
    # but still parses targeted patterns
    assert parse_letter("The answer is D. D-level explanation follows") == "D"
