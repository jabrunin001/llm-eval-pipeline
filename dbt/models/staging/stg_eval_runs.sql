with source as (
    select * from {{ source('raw', 'raw_eval_runs') }}
)

select
    run_id,
    model,
    model_provider,
    prompt_version,
    seed,
    subset_size,
    temperature,
    started_at,
    finished_at,
    status,
    error_message,
    datediff('millisecond', started_at, finished_at) as duration_ms
from source
