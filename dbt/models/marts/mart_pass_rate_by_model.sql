with scored as (
    select * from {{ ref('int_scored_responses') }}
),
agg as (
    select
        model,
        subject_group,
        count(*) as n_responses,
        sum(case when is_correct_strict then 1 else 0 end) as n_correct,
        count(distinct run_id) as n_runs
    from scored
    group by 1, 2
)

select
    model,
    subject_group,
    n_responses,
    n_runs,
    n_correct::double / nullif(n_responses, 0) as pass_rate,
    {{ wilson_ci('n_correct', 'n_responses') }}
from agg
