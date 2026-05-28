{# Wilson 95% confidence interval for a binomial proportion.

   Usage: {{ wilson_ci(successes_col='n_correct', trials_col='n_responses') }}
   Returns two columns: ci_lower, ci_upper.
#}

{% macro wilson_ci(successes_col, trials_col, z=1.96) -%}
    case when {{ trials_col }} = 0 then null else
        (
            ({{ successes_col }}::double / {{ trials_col }})
            + ({{ z }} * {{ z }}) / (2.0 * {{ trials_col }})
            - {{ z }} * sqrt(
                ({{ successes_col }}::double / {{ trials_col }})
                * (1 - {{ successes_col }}::double / {{ trials_col }})
                / {{ trials_col }}
                + ({{ z }} * {{ z }}) / (4.0 * {{ trials_col }} * {{ trials_col }})
            )
        ) / (1 + ({{ z }} * {{ z }}) / {{ trials_col }})
    end as ci_lower,
    case when {{ trials_col }} = 0 then null else
        (
            ({{ successes_col }}::double / {{ trials_col }})
            + ({{ z }} * {{ z }}) / (2.0 * {{ trials_col }})
            + {{ z }} * sqrt(
                ({{ successes_col }}::double / {{ trials_col }})
                * (1 - {{ successes_col }}::double / {{ trials_col }})
                / {{ trials_col }}
                + ({{ z }} * {{ z }}) / (4.0 * {{ trials_col }} * {{ trials_col }})
            )
        ) / (1 + ({{ z }} * {{ z }}) / {{ trials_col }})
    end as ci_upper
{%- endmacro %}
