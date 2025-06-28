#!/bin/bash
# Exit on error
set -e

# Use local Homebrew installation in project directory (for Render compatibility)
export HOMEBREW_PREFIX="$PWD/.homebrew"
export PATH="$PWD/.homebrew/bin:$PATH"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt

# Check if Ollama is already running on its default port
if ! curl -s --head http://127.0.0.1:11434 > /dev/null; then
    echo "Ollama not detected. Starting Ollama server in the background..."
    ollama serve &
    
    # Wait for Ollama to start
    echo "Waiting for Ollama to start..."
    while ! curl -s --head http://127.0.0.1:11434 > /dev/null; do
        sleep 1
    done
    echo "Ollama is up!"
else
    echo "Ollama is already running, proceeding..."
fi

# Set a default port for local execution if $PORT is not provided
export PORT=${PORT:-5001}

# Start the Flask application
echo "Starting Flask server on port $PORT..."
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 600 --max-requests 1000 