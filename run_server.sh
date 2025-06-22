#!/bin/bash
# Exit on error
set -e

# Start Ollama server in the background
ollama serve &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
while ! curl -s --head http://127.0.0.1:11434 > /dev/null; do
    sleep 1
done
echo "Ollama is up!"

# Start the Flask application
echo "Starting Flask server..."
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 600 --max-requests 1000 