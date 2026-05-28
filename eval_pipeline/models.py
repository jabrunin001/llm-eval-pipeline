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
