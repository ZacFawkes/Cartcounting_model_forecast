#!/usr/bin/env bash
# Start the Cart Forecasting API. Run from the api/ directory.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${CART_API_HOST:-127.0.0.1}"
PORT="${CART_API_PORT:-8000}"
PYTHON="${PYTHON:-../.venv/bin/python}"
echo "Starting Cart Forecasting API on http://${HOST}:${PORT} (docs at /docs)"
exec "$PYTHON" -m uvicorn server.app:app --host "$HOST" --port "$PORT" "$@"
