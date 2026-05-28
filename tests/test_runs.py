from eval_pipeline.runs import finish_run, get_run, mark_run_failed, start_run
from eval_pipeline.warehouse import bootstrap_schema, connect


def test_start_run_inserts_partial(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    bootstrap_schema(con)
    rid = start_run(con, model="claude-haiku-4-5", prompt_version="sha1",
                    seed=42, subset_size=10, temperature=0.0)
    run = get_run(con, rid)
    assert run.status == "partial"
    assert run.finished_at is None
    assert run.model == "claude-haiku-4-5"


def test_finish_run_completes(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    bootstrap_schema(con)
    rid = start_run(con, model="m", prompt_version="s", seed=1, subset_size=1, temperature=0.0)
    finish_run(con, rid, status="completed")
    run = get_run(con, rid)
    assert run.status == "completed"
    assert run.finished_at is not None


def test_mark_run_failed_records_error(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    bootstrap_schema(con)
    rid = start_run(con, model="m", prompt_version="s", seed=1, subset_size=1, temperature=0.0)
    mark_run_failed(con, rid, "API quota exceeded")
    run = get_run(con, rid)
    assert run.status == "failed"
    assert run.error_message == "API quota exceeded"
