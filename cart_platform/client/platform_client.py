"""HTTP client for the fake callers. Not deployed with the server."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CLIENT_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = CLIENT_ROOT.parent
PROJECT_ROOT = PLATFORM_ROOT.parent
DATA_FILE = PROJECT_ROOT / "cart_forecasting" / "data" / "real_data.csv"
CLIENT_LOG_DIR = CLIENT_ROOT / "client_logs"
CLIENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE_URL = "http://127.0.0.1:8100"
LOGGER = logging.getLogger("fake_client")


def configure_logging(level: int = logging.INFO) -> None:
    LOGGER.setLevel(level)
    if LOGGER.handlers:
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | client | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(CLIENT_LOG_DIR / "client.log")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    LOGGER.propagate = False


class PlatformClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, name: str = "fake-client",
                 timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self.session = requests.Session()
        self.call_log = CLIENT_LOG_DIR / "client_calls.jsonl"

    def _call(self, method: str, path: str, **kwargs) -> dict:
        t0 = time.perf_counter()
        error, status, body = None, None, {}
        try:
            r = self.session.request(method, f"{self.base_url}{path}",
                                     timeout=self.timeout, **kwargs)
            status = r.status_code
            try:
                body = r.json()
            except ValueError:
                body = {"raw": r.text[:500]}
            if status >= 400:
                error = body.get("detail", f"HTTP {status}")
        except requests.RequestException as exc:
            error = str(exc)
        dt = (time.perf_counter() - t0) * 1000
        with open(self.call_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "client": self.name, "method": method, "path": path,
                "status_code": status, "latency_ms": round(dt, 2),
                "server_request_id": body.get("request_id") if isinstance(body, dict) else None,
                "error": error,
            }) + "\n")
        if error:
            raise RuntimeError(f"{method} {path} failed: {error}")
        return body

    # ------------------------------------------------------------- endpoints
    def health(self):
        return self._call("GET", "/health")

    def model_info(self):
        return self._call("GET", "/v1/model/info")

    def bootstrap(self):
        return self._call("POST", "/v1/registry/bootstrap")

    def entities(self, **params):
        return self._call("GET", "/v1/entities", params=params)

    def entity(self, entity_id: str):
        return self._call("GET", f"/v1/entities/{entity_id}")

    def put_entity(self, payload: dict):
        return self._call("PUT", "/v1/entities", json=payload)

    def put_site(self, payload: dict):
        return self._call("PUT", "/v1/sites", json=payload)

    def score(self, payload: dict):
        return self._call("POST", "/v1/score", json=payload)

    def backtest(self, payload: dict):
        return self._call("POST", "/v1/backtest", json=payload)

    def sessions(self, **params):
        return self._call("GET", "/v1/sessions", params=params)

    def reset(self, entity_id: str | None = None):
        return self._call("POST", "/v1/sessions/reset", json={"entity_id": entity_id})

    def orders(self, **params):
        return self._call("GET", "/v1/orders", params=params)

    def execute_order(self, order_id: str, executed_by: str, actual_carts: int,
                      notes: str | None = None):
        return self._call("POST", f"/v1/orders/{order_id}/execute",
                          json={"executed_by": executed_by, "actual_carts": actual_carts,
                                "notes": notes})

    def logs(self, kind: str = "requests", **params):
        return self._call("GET", f"/v1/logs/{kind}", params=params)

    def wait_until_ready(self, attempts: int = 30, delay: float = 1.0) -> bool:
        for _ in range(attempts):
            try:
                h = self.health()
                if h.get("model_loaded"):
                    LOGGER.info("Server ready — model %s, %d entities (%d active)",
                                h["champion_model_id"], h["n_entities"], h["n_active"])
                    return True
                LOGGER.warning("Server up but no model; bootstrapping…")
                self.bootstrap()
                continue
            except RuntimeError:
                time.sleep(delay)
        LOGGER.error("Server did not become ready")
        return False


# --------------------------------------------------------------------------
# Telemetry helpers
# --------------------------------------------------------------------------
def load_telemetry(path: Path = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df["DVRDate"] = pd.to_datetime(df["DVRDate"])
    df["next_DVRDate"] = pd.to_datetime(df["next_DVRDate"])
    # Identity must match the platform: "siteKey:KDVR:ChanNo" unless the
    # deployment is single-site.
    from settings import INCLUDE_SITE_IN_ENTITY_ID

    parts = [df["KDVR"].astype(str), df["ChanNo"].astype(str)]
    if INCLUDE_SITE_IN_ENTITY_ID and "siteKey" in df.columns:
        parts.insert(0, df["siteKey"].astype(str))
    df["entity_id"] = parts[0].str.cat(parts[1:], sep=":")
    return df.sort_values(["entity_id", "DVRDate"]).reset_index(drop=True)


def to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "DVRDate": r.DVRDate.isoformat(),
            "next_DVRDate": r.next_DVRDate.isoformat(),
            "Count": float(r.Count),
            "Count_IN": float(r.Count_IN),
            "Count_OUT": float(r.Count_OUT),
        }
        for r in df.itertuples()
    ]


def slice_window(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["DVRDate"] >= pd.Timestamp(start)]
    if end:
        out = out[out["DVRDate"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)
