"""Shared HTTP client for the fake API callers.

This package is NOT deployed with the server. It simulates the systems that
would call the API in production: a store controller streaming telemetry to
/score, and an analyst submitting windows to /backtest.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CLIENT_ROOT = Path(__file__).resolve().parent
API_ROOT = CLIENT_ROOT.parent
PROJECT_ROOT = API_ROOT.parent
DATA_FILE = PROJECT_ROOT / "cart_forecasting" / "data" / "real_data.csv"
CLIENT_LOG_DIR = CLIENT_ROOT / "client_logs"
CLIENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

LOGGER = logging.getLogger("fake_client")


def configure_logging(level: int = logging.INFO) -> None:
    LOGGER.setLevel(level)
    if LOGGER.handlers:
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | client | %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(CLIENT_LOG_DIR / "client.log")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


class CartApiClient:
    """Thin requests wrapper that logs every call from the caller's side."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, name: str = "fake-client",
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self.session = requests.Session()
        self.call_log = CLIENT_LOG_DIR / "client_calls.jsonl"

    # ------------------------------------------------------------------ core
    def _call(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        t0 = time.perf_counter()
        error = None
        status = None
        body: dict = {}
        try:
            r = self.session.request(method, url, timeout=self.timeout, **kwargs)
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

        self._log_call(method, path, status, dt, error, body)
        if error:
            raise RuntimeError(f"{method} {path} failed: {error}")
        return body

    def _log_call(self, method, path, status, latency_ms, error, body) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "client": self.name,
            "method": method,
            "path": path,
            "status_code": status,
            "latency_ms": round(latency_ms, 2),
            "server_request_id": body.get("request_id") if isinstance(body, dict) else None,
            "error": error,
        }
        with open(self.call_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # --------------------------------------------------------------- endpoints
    def health(self) -> dict:
        return self._call("GET", "/health")

    def model_info(self) -> dict:
        return self._call("GET", "/model/info")

    def score(self, payload: dict) -> dict:
        return self._call("POST", "/score", json=payload)

    def backtest(self, payload: dict) -> dict:
        return self._call("POST", "/backtest", json=payload)

    def sessions(self) -> dict:
        return self._call("GET", "/sessions")

    def reset_sessions(self, session_id: str | None = None) -> dict:
        return self._call("POST", "/sessions/reset", json={"session_id": session_id})

    def logs(self, kind: str = "requests", limit: int = 50) -> dict:
        return self._call("GET", f"/logs/{kind}", params={"limit": limit})

    def wait_until_ready(self, attempts: int = 30, delay: float = 1.0) -> bool:
        for i in range(attempts):
            try:
                h = self.health()
                if h.get("model_loaded"):
                    LOGGER.info("Server ready (model %s)", h.get("model_version"))
                    return True
                LOGGER.warning("Server up but model not loaded")
                return False
            except RuntimeError:
                time.sleep(delay)
        LOGGER.error("Server did not become ready after %d attempts", attempts)
        return False


# --------------------------------------------------------------------------
# Telemetry helpers
# --------------------------------------------------------------------------
def load_telemetry(path: Path = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df["DVRDate"] = pd.to_datetime(df["DVRDate"])
    df["next_DVRDate"] = pd.to_datetime(df["next_DVRDate"])
    return df.sort_values("DVRDate").reset_index(drop=True)


def to_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "DVRDate": r["DVRDate"].isoformat(),
                "next_DVRDate": r["next_DVRDate"].isoformat(),
                "Count": float(r["Count"]),
                "Count_IN": float(r["Count_IN"]),
                "Count_OUT": float(r["Count_OUT"]),
                "siteKey": int(r["siteKey"]),
            }
        )
    return out


def slice_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["DVRDate"] >= pd.Timestamp(start)]
    if end:
        out = out[out["DVRDate"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)
