#!/bin/bash

# This script automates the setup process for the Dynamic Web Page Generator.

echo "--- Starting Project Setup ---"

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 could not be found. Please install Python 3."
    exit 1
fi

# 2. Create a virtual environment
echo "[1/3] Creating Python virtual environment in 'venv'..."
python3 -m venv venv

# 3. Activate the virtual environment and install dependencies
echo "[2/3] Activating virtual environment and installing dependencies from requirements.txt..."
source venv/bin/activate
pip install -r requirements.txt

# 4. Remind user to install Ollama model
echo "[3/3] Setup complete!"
echo ""
echo "--- IMPORTANT ---"
echo "This project requires the Ollama 'mistral' model."
echo "Please make sure Ollama is running and pull the model by running:"
echo "ollama pull mistral"
echo "-----------------"
echo ""
echo "You can now run the application with: python app.py"

# Install Python dependencies
pip install -r requirements.txt

# Use local Homebrew installation in project directory (for Render compatibility)
export HOMEBREW_PREFIX="$PWD/.homebrew"
export PATH="$PWD/.homebrew/bin:$PATH"

# Install Homebrew locally if not already installed
if [ ! -d "$PWD/.homebrew" ]; then
  echo "Installing Homebrew locally..."
  git clone --depth=1 https://github.com/Homebrew/brew $PWD/.homebrew
fi

# Install Ollama using Homebrew (if not already installed)
if ! command -v ollama &> /dev/null; then
  echo "Installing Ollama with Homebrew..."
  brew install ollama
fi

# Pull the mistral model
ollama serve &
sleep 3  # Give the server a moment to start
ollama pull mistral 