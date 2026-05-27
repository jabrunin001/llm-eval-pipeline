with source as (
    select * from {{ source('raw', 'raw_eval_responses') }}
)

select
    response_id,
    run_id,
    question_id,
    raw_completion,
    parsed_answer,
    is_correct,
    coalesce(is_correct, false) as is_correct_strict,
    (parsed_answer is null) as is_unparseable,
    latency_ms,
    input_tokens,
    output_tokens,
    api_error,
    responded_at
from source
