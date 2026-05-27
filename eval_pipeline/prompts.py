import re

from eval_pipeline.models import Question

PROMPT_TEMPLATE_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are taking a multiple-choice exam. For each question, respond with "
    "ONLY the single letter (A, B, C, or D) of the correct answer. "
    "Do not explain, do not add punctuation, just the letter."
)


def render_prompt(q: Question) -> str:
    return (
        f"Question: {q.question}\n\n"
        f"A) {q.choice_a}\n"
        f"B) {q.choice_b}\n"
        f"C) {q.choice_c}\n"
        f"D) {q.choice_d}\n\n"
        "Answer:"
    )


_LETTER_PATTERNS = [
    # "Answer: X" or "Answer X" with optional parens/colon
    re.compile(r"\bAnswer\s*[:\-]?\s*\*?\*?\(?([A-D])\)?\*?\*?(?:\b|$)", re.IGNORECASE),
    # "answer is X" / "answer is **X**" / "answer is (X)"
    re.compile(r"\banswer is\s*\*?\*?\(?([A-D])\)?\*?\*?(?:\b|$)", re.IGNORECASE),
    # **X** bold
    re.compile(r"\*\*([A-D])\*\*"),
    # (X) parenthesized
    re.compile(r"\(([A-D])\)", re.IGNORECASE),
    # Letter followed by period or close-paren ("A." or "A)")
    re.compile(r"\b([A-D])(?=[\.\)])", re.IGNORECASE),
    # Last resort: standalone letter at word boundary, not followed by a hyphen
    # (only used if exactly one such letter appears in the response)
    re.compile(r"\b([A-D])(?!-)\b", re.IGNORECASE),
]


def parse_letter(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    # Try targeted patterns first (specific cues for the answer)
    for pat in _LETTER_PATTERNS[:-1]:  # all except last-resort
        m = pat.search(raw)
        if m:
            return m.group(1).upper()
    # Last-resort: only fire if there is exactly one A-D letter at word boundaries
    last_resort = _LETTER_PATTERNS[-1]
    matches = {m.group(1).upper() for m in last_resort.finditer(raw)}
    if len(matches) == 1:
        return matches.pop()
    return None
