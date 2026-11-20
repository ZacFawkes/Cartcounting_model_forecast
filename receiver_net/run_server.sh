#!/usr/bin/env bash
# Cart Result Receiver (.NET). Mirrors cart_platform/receiver/run_server.sh.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${CART_RECEIVER_HOST:-127.0.0.1}"
PORT="${CART_RECEIVER_PORT:-8200}"

if [[ -z "${CART_RECEIVER_SQL_CONNECTION:-}" ]]; then
  echo "CART_RECEIVER_SQL_CONNECTION is not set." >&2
  echo "Example: export CART_RECEIVER_SQL_CONNECTION=\\" >&2
  echo "  'Server=localhost,1433;Database=CartResults;User Id=cart_receiver;Password=***;Encrypt=True;TrustServerCertificate=True'" >&2
  exit 1
fi

echo "Cart Result Receiver (.NET) -> http://${HOST}:${PORT}/docs"
exec dotnet run --project src/ReceiverNet/ReceiverNet.csproj \
  --urls "http://${HOST}:${PORT}" "$@"
