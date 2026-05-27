.PHONY: all install load score build dashboard test lint typecheck docs clean

EVAL_DB_PATH ?= data/eval.duckdb
export EVAL_DB_PATH

all: install load score build
	@echo "Done. Run 'make dashboard' to view."

install:
	uv sync

load:
	uv run python -m eval_pipeline.cli load

score:
	uv run python -m eval_pipeline.cli run

build:
	cd dbt && uv run dbt seed --profiles-dir . && uv run dbt build --profiles-dir .

dashboard:
	uv run streamlit run dashboard/app.py

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
