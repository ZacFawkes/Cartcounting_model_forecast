#!/usr/bin/env bash
# Cart Result Publisher (.NET): durable outbox + retry worker.
# Mirrors the Python integrations/publisher.py worker on the ML server.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${CART_PUBLISHER_HOST:-127.0.0.1}"
PORT="${CART_PUBLISHER_PORT:-8300}"

if [[ -z "${CART_OUTBOX_SQL_CONNECTION:-}" ]]; then
  echo "CART_OUTBOX_SQL_CONNECTION is not set." >&2
  echo "Example:" >&2
  echo "  export CART_OUTBOX_SQL_CONNECTION='Server=localhost,1433;Database=CartResults;User Id=cart_receiver;Password=***;Encrypt=True;TrustServerCertificate=True'" >&2
  exit 1
fi

echo "Cart Result Publisher -> http://${HOST}:${PORT}/docs"
echo "  outbox   : ${CART_OUTBOX_SQL_CONNECTION%%;*}"
echo "  receiver : ${CART_RESULT_RECEIVER_URL:-http://127.0.0.1:8200}"
exec dotnet run --project src/PublisherHost/PublisherHost.csproj \
  --urls "http://${HOST}:${PORT}" "$@"
