# Lineage Diagram

Run `make docs` (or `cd dbt && uv run dbt docs generate --profiles-dir .`), then open `dbt/target/index.html` in a browser. Use the "view lineage" button (bottom-right) to see the full graph.

For a static SVG/PNG:
1. Run `make docs`
2. Open `dbt/target/index.html`
3. Click the lineage button (bottom-right corner)
4. Screenshot the rendered graph

A rendered version will be added to this directory once produced.
