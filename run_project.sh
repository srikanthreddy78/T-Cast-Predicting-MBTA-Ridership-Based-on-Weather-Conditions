#!/bin/bash

set -e

echo "Creating virtual environment..."
python -m venv venv

echo "Activating virtual environment..."
# Detect OS type and activate accordingly
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
  # Git Bash or Windows
  source venv/Scripts/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  source venv/bin/activate
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux
  source venv/bin/activate
else
  echo "Unsupported OS: $OSTYPE"
  exit 1
fi

echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Running preprocessing..."
python src/data/processing.py

echo "Running model pipeline..."
python src/data/tuningModel.py

echo "Running visualization..."
python visualization.py

echo "Project completed successfully!"
