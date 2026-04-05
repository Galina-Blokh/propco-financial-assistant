#!/usr/bin/env bash
# Start MLflow UI and Streamlit together from the repo root.
# Ctrl+C stops Streamlit and tears down MLflow.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
# Match Python default: file store under repo (avoids SQLite lock fights with the UI)
TRACK_DIR="${ROOT}/mlflow-tracking"
mkdir -p "$TRACK_DIR"
export MLFLOW_TRACKING_URI="${MLFLOW_STORE_URI:-file://${TRACK_DIR}}"
MLFLOW_STORE_URI="$MLFLOW_TRACKING_URI"

cleanup() {
  if [[ -n "${MLFLOW_PID:-}" ]] && kill -0 "$MLFLOW_PID" 2>/dev/null; then
    kill "$MLFLOW_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting MLflow UI on port ${MLFLOW_PORT} (${MLFLOW_STORE_URI}) …"
uv run mlflow ui \
  --backend-store-uri "$MLFLOW_STORE_URI" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT" &
MLFLOW_PID=$!

sleep 1
echo "MLflow: http://localhost:${MLFLOW_PORT}"
echo "Starting Streamlit on port ${STREAMLIT_PORT} …"

uv run streamlit run ui/streamlit_app.py \
  --server.headless=true \
  --server.port="$STREAMLIT_PORT"
