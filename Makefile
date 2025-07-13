.PHONY: create-graph search-graph visualize-graph run-agent format-lint-fix

create-graph:
	@echo "Creating graph from files: $(files)"
	uv run src/infrastructure/cognee_pipeline.py --files "$(files)"

search-graph:
	uv run streamlit run src/infrastructure/cognee_search.py

visualize-graph:
	@echo "Starting server at http://localhost:8000"
	cd .artifacts && uv run python -m http.server

run-agent:
	export PYTHONPATH=$(shell pwd)/src:$$PYTHONPATH; uv run streamlit run src/application/main.py

format-lint-fix:
	uv run ruff format .
	uv run ruff check --select I --fix
	uv run ruff check --fix