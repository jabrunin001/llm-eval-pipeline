-- pass_rate in marts must be between 0 and 1.
select 'mart_pass_rate_by_model' as src, pass_rate
from {{ ref('mart_pass_rate_by_model') }}
where pass_rate < 0 or pass_rate > 1
union all
select 'mart_category_breakdown', pass_rate
from {{ ref('mart_category_breakdown') }}
where pass_rate < 0 or pass_rate > 1
union all
select 'mart_run_drift', pass_rate
from {{ ref('mart_run_drift') }}
where pass_rate < 0 or pass_rate > 1
