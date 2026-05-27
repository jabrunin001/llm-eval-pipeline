with source as (
    select * from {{ source('raw', 'raw_mmlu_questions') }}
)

select
    question_id,
    lower(subject) as subject,
    question,
    choice_a,
    choice_b,
    choice_c,
    choice_d,
    answer,
    dataset_version,
    loaded_at
from source
