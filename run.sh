#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uv run streamlit run src/application/main.py