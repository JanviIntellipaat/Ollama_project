#!/usr/bin/env bash
set -euo pipefail

# --- Ollama check ---
if ! command -v ollama &> /dev/null; then
  echo "❌ Ollama not found. Please install Ollama first."
  exit 1
fi

# Start Ollama if not already running
if ! pgrep -x "ollama" > /dev/null; then
  echo "▶️ Starting Ollama daemon..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
else
  echo "✅ Ollama daemon already running."
fi

# Ensure required models are pulled
if ! ollama list | grep -q "nomic-embed-text"; then
  echo "⬇️ Pulling nomic-embed-text model..."
  ollama pull nomic-embed-text
fi
if ! ollama list | grep -q "mistral"; then
  echo "⬇️ Pulling mistral model..."
  ollama pull mistral
fi

# Run Streamlit app
echo "🚀 Launching Streamlit UI..."
exec streamlit run streamlit_app_fixed.py
