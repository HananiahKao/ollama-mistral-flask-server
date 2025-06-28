#!/bin/bash

# This script activates the virtual environment and starts the Flask server.
# It should be run from the root of the project directory.

echo "--- Starting Application Server ---"

# Use local Homebrew installation in project directory (for Render compatibility)
export HOMEBREW_PREFIX="$PWD/.homebrew"
export PATH="$PWD/.homebrew/bin:$PATH"

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment 'venv' not found."
    echo "Please run the ./setup.sh script first."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Run the python application
python app.py 