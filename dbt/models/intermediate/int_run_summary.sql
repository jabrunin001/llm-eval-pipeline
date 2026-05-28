with scored as (
    select * from {{ ref('int_scored_responses') }}
)

select
    run_id,
    model,
    prompt_version,
    run_started_at,
    count(*) as n_responses,
    sum(case when is_unparseable then 0 else 1 end) as n_parsed,
    sum(case when is_correct_strict then 1 else 0 end) as n_correct,
    avg(case when is_correct_strict then 1.0 else 0.0 end) as pass_rate,
    avg(case when is_unparseable then 1.0 else 0.0 end) as unparseable_rate,
    avg(latency_ms) as mean_latency_ms,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens
from scored
group by 1, 2, 3, 4
