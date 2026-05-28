.PHONY: all install load score build dashboard test lint typecheck docs clean

EVAL_DB_PATH ?= data/eval.duckdb
export EVAL_DB_PATH

all: install load score build export-data
	@echo "Done. Run 'make dashboard' to view the Streamlit app, or 'make static-dashboard' to view the premium web dashboard."

install:
	uv sync

load:
	uv run python -m eval_pipeline.cli load

score:
	uv run python -m eval_pipeline.cli run

build:
	cd dbt && uv run dbt seed --profiles-dir . && uv run dbt build --profiles-dir .

export-data:
	uv run python scripts/export_dashboard_data.py

dashboard:
	uv run streamlit run dashboard/app.py

static-dashboard: export-data
	@echo "========================================================================"
	@echo "Launching Premium Web Dashboard (GitHub Pages Compatible)"
	@echo "To preview, open docs/index.html in your browser, or visit:"
	@echo "http://localhost:8000"
	@echo "========================================================================"
	uv run python -m http.server 8000 --directory docs

test:
	uv run pytest -m "not integration" -v

lint:
	uv run ruff check .

typecheck:
	uv run mypy eval_pipeline

docs:
	cd dbt && uv run dbt docs generate --profiles-dir .
	@echo "dbt docs in dbt/target/. Open dbt/target/index.html in a browser."

clean:
	rm -rf data/*.duckdb data/*.duckdb.wal dbt/target dbt/dbt_packages
