with responses as (
    select * from {{ ref('stg_eval_responses') }}
),
runs as (
    select * from {{ ref('stg_eval_runs') }}
),
questions as (
    select * from {{ ref('stg_mmlu_questions') }}
),
groups as (
    select * from {{ ref('mmlu_subject_groups') }}
)

select
    r.response_id,
    r.run_id,
    r.question_id,
    ru.model,
    ru.prompt_version,
    ru.seed,
    ru.started_at as run_started_at,
    q.subject,
    coalesce(g.subject_group, 'other') as subject_group,
    q.answer as ground_truth,
    r.parsed_answer,
    r.is_correct,
    r.is_correct_strict,
    r.is_unparseable,
    r.latency_ms,
    r.input_tokens,
    r.output_tokens,
    r.api_error
from responses r
join runs ru using (run_id)
join questions q using (question_id)
left join groups g using (subject)
where ru.status = 'completed'
