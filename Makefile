create-graph:
	@echo "Crheating graph from files: $(files)"
	uv run src/infrastructure/cognee_pipeline.py --files "$(files)"

search-graph:
	uv run streamlit run src/infrastructure/cognee_search.py

run-agent:
	export PYTHONPATH=$(shell pwd)/src:$$PYTHONPATH; uv run streamlit run src/application/main.py

format-lit-fix:
	uv run ruff format .
	uv run ruff check --select I --fix
	uv run ruff check --fix