#!/bin/bash
set -e

# Use Python 3.9
export PYTHON_VERSION=3.9.19

# Upgrade pip, setuptools, wheel first
pip install --upgrade pip setuptools wheel

# Install all dependencies without build isolation
pip install --no-build-isolation -r requirements.txt

echo "Build complete!"
