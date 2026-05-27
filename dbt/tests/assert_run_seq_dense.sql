-- run_seq must be dense (1, 2, 3, ...) per model — no gaps.
with seq as (
    select model, run_seq,
           lag(run_seq) over (partition by model order by run_seq) as prev_seq
    from {{ ref('mart_run_drift') }}
)
select model, run_seq, prev_seq
from seq
where prev_seq is not null and run_seq <> prev_seq + 1
