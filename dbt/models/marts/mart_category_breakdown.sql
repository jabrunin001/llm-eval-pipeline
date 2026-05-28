with scored as (
    select * from {{ ref('int_scored_responses') }}
)

select
    model,
    subject,
    count(*) as n_responses,
    count(distinct run_id) as n_runs,
    sum(case when is_correct_strict then 1 else 0 end)::double
        / nullif(count(*), 0) as pass_rate,
    avg(case when is_unparseable then 1.0 else 0.0 end) as unparseable_rate
from scored
group by 1, 2
