#!/usr/bin/env bash
# Start the Cart Forecasting Platform API. Run from cart_platform/.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${CART_PLATFORM_HOST:-127.0.0.1}"
PORT="${CART_PLATFORM_PORT:-8100}"
PYTHON="${PYTHON:-../.venv/bin/python}"
echo "Cart Forecasting Platform → http://${HOST}:${PORT}/docs"
exec "$PYTHON" -m uvicorn server.app:app --host "$HOST" --port "$PORT" "$@"
