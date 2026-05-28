-- Every raw_eval_responses.run_id must exist in raw_eval_runs.
select r.run_id
from {{ source('raw', 'raw_eval_responses') }} r
left join {{ source('raw', 'raw_eval_runs') }} ru using (run_id)
where ru.run_id is null
