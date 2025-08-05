#!/bin/bash

# Install correct package
pip install google-generativeai

# Create alias for agno (if needed)
pip install --upgrade pip
pip install --no-deps --force-reinstall git+https://github.com/google/generative-ai-python

# Run your app
streamlit run app.py
