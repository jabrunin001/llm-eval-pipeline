from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

st.set_page_config(page_title="LLM Eval Pipeline", layout="wide")
st.title("LLM Eval Pipeline — MMLU vs Claude")

DB_PATH = Path(os.getenv("EVAL_DB_PATH", "data/eval.duckdb"))


@st.cache_data(ttl=60)
def query(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


if not DB_PATH.exists():
    st.error(f"No DuckDB at {DB_PATH}. Run `make all` or `python -m eval_pipeline.cli run`.")
    st.stop()

st.header("Pass rate by model (with 95% Wilson CIs)")
df_pass = query("SELECT * FROM mart_pass_rate_by_model ORDER BY model, subject_group")
if df_pass.empty:
    st.warning("No data in mart_pass_rate_by_model. Run scoring + dbt build first.")
else:
    pivot = df_pass.pivot(index="subject_group", columns="model", values="pass_rate")
    st.bar_chart(pivot)
    with st.expander("Raw mart_pass_rate_by_model"):
        st.dataframe(df_pass, use_container_width=True)

st.header("Category breakdown (top + bottom 5 categories per model)")
df_cat = query("SELECT * FROM mart_category_breakdown ORDER BY model, pass_rate")
if not df_cat.empty:
    models = sorted(df_cat["model"].unique())
    cols = st.columns(len(models))
    for col, m in zip(cols, models, strict=False):
        sub = df_cat[df_cat["model"] == m]
        col.subheader(m)
        col.dataframe(
            pd.concat([sub.head(5), sub.tail(5)])[["subject", "pass_rate", "unparseable_rate"]],
            use_container_width=True,
        )

st.header("Drift across runs")
df_drift = query("SELECT * FROM mart_run_drift ORDER BY model, run_seq")
if not df_drift.empty:
    chart = df_drift.pivot(index="run_seq", columns="model", values="pass_rate")
    st.line_chart(chart)
    with st.expander("delta_from_first_run"):
        st.dataframe(
            df_drift[["model", "run_seq", "pass_rate", "delta_from_first_run"]],
            use_container_width=True,
        )

st.caption(f"Source: {DB_PATH} • Refreshed every 60s")
