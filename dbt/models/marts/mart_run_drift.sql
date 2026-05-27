with summary as (
    select * from {{ ref('int_run_summary') }}
),
ordered as (
    select
        model,
        run_id,
        run_started_at,
        pass_rate,
        n_responses,
        row_number() over (partition by model order by run_started_at) as run_seq
    from summary
),
first_run as (
    select model, pass_rate as first_pass_rate
    from ordered
    where run_seq = 1
)

select
    o.model,
    o.run_id,
    o.run_seq,
    o.run_started_at,
    o.pass_rate,
    o.n_responses,
    o.pass_rate - fr.first_pass_rate as delta_from_first_run
from ordered o
left join first_run fr using (model)
order by o.model, o.run_seq
