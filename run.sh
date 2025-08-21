#!/usr/bin/env bash
set -euo pipefail

# ========= Corp proxy (adjust if needed) =========
export HTTP_PROXY="${HTTP_PROXY:-http://webproxy.deutsche-boerse.de:8080}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"

# ========= Ollama: user-owned daemon on 11435 =========
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11435}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435/api}"
LOG="/tmp/ollama_user_11435.log"

# ========= Streamlit bind (tunnel-friendly) =========
STREAMLIT_ADDRESS="${STREAMLIT_ADDRESS:-127.0.0.1}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Missing '$1'"; exit 1; }; }
need_cmd ollama
need_cmd curl
need_cmd streamlit

echo "👉 Proxy: $HTTP_PROXY"
echo "👉 Ollama: $OLLAMA_HOST  (API: $OLLAMA_BASE_URL)"
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
    echo "✅ Ollama daemon responding on $OLLAMA_HOST"
    return 0
  fi
  echo "▶️ Starting user ollama daemon on $OLLAMA_HOST ..."
  nohup ollama serve >"$LOG" 2>&1 &
  sleep 2
  if wait_for_api; then
    echo "✅ Ollama API up on $OLLAMA_HOST"
  else
    echo "❌ Ollama failed to start on $OLLAMA_HOST"
    tail -n 80 "$LOG" || true
    exit 1
  fi
}

ensure_model() {
  local model="$1"
  if curl -sf "$OLLAMA_BASE_URL/tags" | grep -q "\"$model\""; then
    echo "✅ Model present: $model"
  else
    echo "⬇️ Pulling model: $model"
    if ! ollama pull "$model"; then
      echo "❌ Failed to pull model: $model"
      tail -n 80 "$LOG" || true
      exit 1
    fi
  fi
}

# 1) Start Ollama (proxy-aware)
start_daemon_if_needed

# 2) Ensure required models
ensure_model "nomic-embed-text"
ensure_model "mistral"

# 3) Pick Streamlit app file
APP="streamlit_app.py"
[[ -f "streamlit_app_fixed.py" ]] && APP="streamlit_app_fixed.py"

# 4) Launch Streamlit with tunnel-friendly flags
echo "🚀 Launching Streamlit: $APP"
exec streamlit run "$APP" \
  --server.address "$STREAMLIT_ADDRESS" \
  --server.port "$STREAMLIT_PORT" \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
