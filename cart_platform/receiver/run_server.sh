#!/usr/bin/env bash
# Standalone result receiver API. Run from cart_platform/.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${CART_RECEIVER_HOST:-127.0.0.1}"
PORT="${CART_RECEIVER_PORT:-8200}"
PYTHON="${PYTHON:-../.venv/bin/python}"
echo "Cart Result Receiver → http://${HOST}:${PORT}/docs"
exec "$PYTHON" -m uvicorn receiver.app:app --host "$HOST" --port "$PORT" "$@"
