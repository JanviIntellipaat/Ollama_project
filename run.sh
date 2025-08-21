#!/usr/bin/env bash
set -euo pipefail

# ========= Proxy (override if needed) =========
# Use your existing env if set; otherwise fall back to the DB proxy.
export HTTP_PROXY="${HTTP_PROXY:-http://webproxy.deutsche-boerse.de:8080}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"

# ========= User-scoped Ollama paths =========
# Keep models in your home to avoid permission clashes with other users
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
mkdir -p "$OLLAMA_MODELS" 2>/dev/null || true

# Logs in your user cache dir (fixes /tmp permission issues across users)
LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/ollama"
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG="$LOG_DIR/ollama_${USER}_11435.log"

# ========= Ollama server (user-owned) =========
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11435}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435/api}"

# ========= Streamlit bind (tunnel-friendly) =========
STREAMLIT_ADDRESS="${STREAMLIT_ADDRESS:-127.0.0.1}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Missing '$1'"; exit 1; }; }
need_cmd ollama
need_cmd curl
need_cmd streamlit

echo "👉 Proxy: $HTTP_PROXY"
echo "👉 Ollama: $OLLAMA_HOST  (API: $OLLAMA_BASE_URL)"
echo "👉 Models dir: $OLLAMA_MODELS"
echo "👉 Log: $LOG"
echo "👉 Streamlit: $STREAMLIT_ADDRESS:$STREAMLIT_PORT"

wait_for_api() {
  local tries=0
  while (( tries < 30 )); do
    if curl -sf "$OLLAMA_BASE_URL/tags" >/dev/null; then return 0; fi
    sleep 1; tries=$((tries+1))
  done
  return 1
}

start_daemon_if_needed() {
  if curl -sf "$OLLAMA_BASE_URL/tags" >/dev/null; then
    echo "✅ Ollama daemon already responding on $OLLAMA_HOST"
    return 0
  fi

  echo "▶️ Starting user ollama daemon on $OLLAMA_HOST ..."
  # Important: nohup writes to a user-writable log file
  nohup ollama serve >"$LOG" 2>&1 &
  sleep 2

  if wait_for_api; then
    echo "✅ Ollama API up on $OLLAMA_HOST"
  else
    echo "❌ Ollama failed to start on $OLLAMA_HOST"
    echo "── Log tail ─────────────────────────────────────────"
    tail -n 120 "$LOG" || true
    echo "────────────────────────────────────────────────────"
    exit 1
  fi
}

ensure_model() {
  local model="$1"
  # Query the daemon for installed models
  if curl -sf "$OLLAMA_BASE_URL/tags" | grep -q "\"$model\""; then
    echo "✅ Model present: $model"
  else
    echo "⬇️ Pulling model: $model"
    if ! ollama pull "$model"; then
      echo "❌ Failed to pull model: $model"
      echo "── Log tail ─────────────────────────────────────────"
      tail -n 120 "$LOG" || true
      echo "────────────────────────────────────────────────────"
      exit 1
    fi
  fi
}

# 1) Start Ollama (user-owned, proxy-aware)
start_daemon_if_needed

# 2) Ensure required models
ensure_model "nomic-embed-text"
ensure_model "mistral"

# 3) Choose app file
APP="streamlit_app.py"
[[ -f "streamlit_app_fixed.py" ]] && APP="streamlit_app_fixed.py"

# 4) Launch Streamlit with tunnel-friendly flags
echo "🚀 Launching Streamlit: $APP"
exec streamlit run "$APP" \
  --server.address "$STREAMLIT_ADDRESS" \
  --server.port "$STREAMLIT_PORT" \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
