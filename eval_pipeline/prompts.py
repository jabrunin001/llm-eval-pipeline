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
