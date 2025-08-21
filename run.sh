cat > run.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# ========= Corp proxy (adjust if needed) =========
export HTTP_PROXY="http://webproxy.deutsche-boerse.de:8080"
export HTTPS_PROXY="$HTTP_PROXY"
export NO_PROXY="127.0.0.1,localhost,::1"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"

# ========= Ollama: run a user-owned daemon on 11435 =========
export OLLAMA_HOST="127.0.0.1:11435"
export OLLAMA_BASE_URL="http://127.0.0.1:11435/api"
LOG="/tmp/ollama_user_11435.log"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "❌ Required command '$1' not found. Install it and retry."; exit 1; }
}

need_cmd ollama
need_cmd curl
need_cmd streamlit

echo "👉 Using proxy: $HTTP_PROXY"
echo "👉 Ollama host: $OLLAMA_HOST"
echo "👉 API URL    : $OLLAMA_BASE_URL"

wait_for_api() {
  local tries=0
  while (( tries < 30 )); do
    if curl -sf "$OLLAMA_BASE_URL/tags" >/dev/null; then
      return 0
    fi
    sleep 1
    tries=$((tries+1))
  done
  return 1
}

start_daemon_if_needed() {
  if curl -sf "$OLLAMA_BASE_URL/tags" >/dev/null; then
    echo "✅ Ollama daemon already responding on $OLLAMA_HOST"
    return 0
  fi

  echo "▶️ Starting user ollama daemon on $OLLAMA_HOST ..."
  nohup ollama serve >"$LOG" 2>&1 &
  sleep 2

  if wait_for_api; then
    echo "✅ Ollama API up on $OLLAMA_HOST"
  else
    echo "❌ Ollama failed to start or not reachable on $OLLAMA_HOST"
    echo "   Log tail:"
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
      echo "   Check network/proxy and the daemon log:"
      tail -n 80 "$LOG" || true
      exit 1
    fi
  fi
}

# 1) Start daemon (user-owned, proxy-aware)
start_daemon_if_needed

# 2) Ensure required models
ensure_model "nomic-embed-text"
ensure_model "mistral"

# 3) Pick Streamlit app file
APP="streamlit_app_fixed.py"
if [[ ! -f "$APP" ]]; then
  if [[ -f "streamlit_app.py" ]]; then
    APP="streamlit_app.py"
  else
    echo "❌ Could not find streamlit_app_fixed.py or streamlit_app.py in $(pwd)"
    exit 1
  fi
fi
echo "🚀 Launching Streamlit: $APP"
exec streamlit run "$APP"
EOF

chmod +x run.sh
