FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 sqlite3 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cart_platform/requirements.txt /tmp/req-platform.txt
COPY cart_forecasting/requirements.txt /tmp/req-pipeline.txt
RUN pip install -r /tmp/req-platform.txt -r /tmp/req-pipeline.txt

COPY cart_forecasting /app/cart_forecasting
COPY cart_platform    /app/cart_platform

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/cart_platform/data/uploads /app/cart_platform/logs /app/cart_platform/models \
 && chown -R app:app /app
USER app

WORKDIR /app/cart_platform

ENV CART_PLATFORM_HOST=0.0.0.0 \
    CART_PLATFORM_PORT=8100 \
    CART_PLATFORM_DB=/app/cart_platform/data/platform.db \
    CART_PLATFORM_UPLOAD_DIR=/app/cart_platform/data/uploads \
    CART_PLATFORM_LOG_DIR=/app/cart_platform/logs \
    CART_RECEIVER_HOST=0.0.0.0 \
    CART_RECEIVER_PORT=8200 \
    CART_RECEIVER_DB=/app/cart_platform/data/results_receiver.db

VOLUME ["/app/cart_platform/data", "/app/cart_platform/logs", "/app/cart_platform/models"]

EXPOSE 8100 8200

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD sh -c 'curl -fsS "http://127.0.0.1:${HEALTH_PORT:-${CART_PLATFORM_PORT}}/health" || exit 1'

CMD ["sh", "-c", "exec uvicorn server.app:app --host \"$CART_PLATFORM_HOST\" --port \"$CART_PLATFORM_PORT\""]
