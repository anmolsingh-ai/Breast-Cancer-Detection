#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 8000 &
# Use the `run` subcommand so Streamlit executes the script file
streamlit run streamlit.py --server.port=8501 --server.address=0.0.0.0
