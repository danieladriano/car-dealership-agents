#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
streamlit run src/application/main.py