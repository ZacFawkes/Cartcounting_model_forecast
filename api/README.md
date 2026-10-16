# Cart Forecasting & Replenishment API

Serves the **validated** forecasting models and the **audited** replenishment
decision engine over HTTP.

```
api/
├── server/                     # DEPLOYED ON THE SERVER
│   ├── app.py                  # FastAPI routes + request logging middleware
│   ├── engine.py               # scoring + backtesting engine
│   ├── schemas.py              # request/response contracts
│   ├── request_log.py          # per-call logging (JSONL + CSV + run artefacts)
│   ├── settings.py             # paths, defaults, served policy
│   └── run_server.sh           # start script
│
├── client/                     # FAKE CALLERS — NOT DEPLOYED
│   ├── api_client.py           # HTTP wrapper + client-side call log
│   ├── fake_score_caller.py    # replays telemetry to /score, record by record
│   ├── fake_backtest_caller.py # submits windows to /backtest under 6 scenarios
│   ├── run_demo.py             # full end-to-end walkthrough
│   └── client_logs/            # caller-side logs (written at runtime)
│
├── logs/                       # SERVER-SIDE LOGS (written at runtime)
│   ├── requests.jsonl          # every call: id, endpoint, latency, outcome
│   ├── score_results.csv       # one row per scoring decision
│   ├── backtest_results.csv    # one row per backtest run
│   ├── errors.jsonl            # rejected / failed calls
│   ├── server.log              # application log
│   └── runs/<run_id>/          # full artefacts of each backtest
│
├── tests/test_api.py           # 18 tests, no running server needed
└── requirements.txt
```

The two folders are independent deployables: `server/` never imports from
`client/`, and `client/` only talks HTTP.

## Quick start

```bash
cd api
pip install -r requirements.txt

# terminal 1 — the server
./server/run_server.sh                 # http://127.0.0.1:8000/docs

# terminal 2 — the fake callers
python client/run_demo.py              # everything
python client/fake_backtest_caller.py  # backtest scenarios only
python client/fake_score_caller.py --start "2026-10-11 08:00" --end "2026-10-11 20:00"
```

Configuration via environment: `CART_API_HOST`, `CART_API_PORT`,
`CART_API_LOG_DIR`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and model status |
| GET | `/model/info` | Models, uncertainty method, served policy, caveats |
| POST | `/score` | One live replenishment decision |
| POST | `/backtest` | Backtest a telemetry window |
| GET | `/sessions` | Online state: pending orders, alert state |
| POST | `/sessions/reset` | Clear online state |
| GET | `/logs/requests` · `/logs/scores` · `/logs/backtests` | Recent call logs |
| GET | `/runs` · `/runs/{run_id}` | Stored backtest artefacts |

### POST /score

Send the newest completed record plus at least **12** consecutive records of
history. The server never sees the future.

```json
{
  "site_key": 163,
  "session_id": "store-163",
  "history": [
    {"DVRDate": "2026-10-11T10:40:00", "next_DVRDate": "2026-10-11T10:46:40",
     "Count": 22, "Count_IN": 18, "Count_OUT": 14}
  ],
  "policy": {"risk_threshold": 0.55}
}
```

Response (abridged):

```json
{
  "decision_time": "2026-10-11T10:53:08",
  "current_cart": 22.0,
  "forecast": {"cart_at_20min": 18.4, "min_within_40min": 12.6,
               "lower_bound_min_within_40min": 6.0},
  "shortage_risk": 0.7634,
  "alert_state": "CRITICAL",
  "action": {"should_replenish": true, "quantity": 16, "urgency": "URGENT",
             "order_id": "ORD-0001",
             "expected_completion_time": "2026-10-11T11:03:08",
             "reason": "risk 0.76 >= 0.55; predicted 40-minute minimum 12.6 carts ..."},
  "pending_carts": 16.0
}
```

`session_id` carries pending orders, cooldown and alert state across calls.
Re-sending a record that was already processed is a no-op — it will not place a
second order.

### POST /backtest

```json
{"use_bundled_data": true, "start": "2026-10-10 00:00", "end": "2026-10-12 23:59",
 "mode": "replay", "policy": {"risk_threshold": 0.55},
 "inventory": {"consumption_model": "unmet_demand"}, "label": "weekly review"}
```

Or post your own `records`. Returns measured `forecast_accuracy`, simulated
`metrics`, `attribution_summary`, the decision list, and a `run_id` whose full
artefacts are kept under `logs/runs/<run_id>/`.

`mode`: `replay` (fast, ~120 ms) or `closed_loop` (rebuilds features from the
simulated state at every step, ~20 s).

## What is served, and what it means

The API serves the **audited re-optimised policy**, not the pre-audit one:

| Parameter | Pre-audit | Served |
|---|---|---|
| Risk threshold | 0.40 | **0.55** |
| Cooldown | 10 min | **30 min** |
| Target stock | 28 | 28 |
| Consumption model | 45-min half-life | **unmet_demand** |
| Reserve | unlimited | **60 carts/day** |

Both endpoints call the same `plan_replenishment()` function, so an online
recommendation can never diverge from what the backtest would have produced.

**Forecast accuracy is measured. Shortage-reduction figures are simulator
output.** The scenarios exposed by `fake_backtest_caller.py` show why that
distinction matters:

| Scenario | Events | Severity reduction | Trips |
|---|---|---|---|
| Audited default | 13 → 2 | 94.6% | 7 |
| Closed-loop | 13 → 4 | 82.5% | 7 |
| Best case | 13 → 1 | 98.3% | 2 |
| **Conservative** | 13 → 12 | **4.2%** | 9 |

Never quote a single reduction percentage to the business.

## Logging

Every call writes three artefacts:

1. `logs/requests.jsonl` — request id, endpoint, latency, client, outcome, error
2. `logs/score_results.csv` or `logs/backtest_results.csv` — flat, spreadsheet-ready
3. `logs/runs/<run_id>/` — for backtests: decisions, order book, events,
   attribution, series, forecasts and the full JSON response

Callers additionally log their own side to `client/client_logs/`
(`client_calls.jsonl`, `fake_score_stream.csv`, `fake_backtest_results.csv`), so
client-observed and server-observed latency can be compared.

Logging never raises: a failed log write cannot fail an API call. Every response
carries `X-Request-ID` and `X-Response-Time-ms` headers.

## Tests

```bash
python -m pytest api/tests -q      # 18 tests, uses TestClient
```

Covers contract shape, insufficient-history rejection, pending-order netting,
lead-time enforcement, session isolation and idempotency, backtest overrides,
assumption sensitivity, artefact retrieval and log integrity.

## Prerequisites

`cart_forecasting/outputs/models/cart_forecasting_models.pkl` must exist. If it
does not:

```bash
cd cart_forecasting && python main.py
```
