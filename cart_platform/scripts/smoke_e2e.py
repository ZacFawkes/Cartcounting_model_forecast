"""End-to-end smoke test against two RUNNING servers.

    Terminal A:  bash server/run_server.sh        # ML  :8100
    Terminal B:  bash receiver/run_server.sh      # receiver :8200
    Terminal C:  python scripts/smoke_e2e.py

Walks the complete workflow: upload -> discover -> train one entity ->
auto-calibration -> three backtests -> manual promotion -> entity scoring ->
CSV artifact -> receiver delivery.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_multi_entity.csv"

CONFIG = {
    "name": "smoke_batch_01",
    "timezone": "Asia/Ho_Chi_Minh",
    "column_mapping": {"KDVR": "KDVR", "ChanNo": "ChanNo", "siteKey": "siteKey",
                       "DVRDate": "DVRDate", "next_DVRDate": "next_DVRDate",
                       "Count": "Count", "Count_IN": "Count_IN",
                       "Count_OUT": "Count_OUT"},
    "entity_defaults": {"max_sequence_gap_min": 15,
                        "cooldown_min": 30, "lead_time_min": 10},
    "entity_overrides": {
        "163:215:0": {"bay_capacity": 60, "red_threshold": 15, "amber_threshold": 20,
                  "target_stock": 28},
        "163:215:1": {"bay_capacity": 90, "red_threshold": 20, "amber_threshold": 28,
                  "target_stock": 40},
    },
    "auto_start_training": False,
}
TARGET = "163:215:0"
BAR = "=" * 78


def step(n, title):
    print(f"\n{BAR}\nSTEP {n} — {title}\n{BAR}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ml", default="http://127.0.0.1:8100")
    p.add_argument("--receiver", default="http://127.0.0.1:8200")
    p.add_argument("--token", default=None)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    H = {"Authorization": f"Bearer {a.token}"} if a.token else {}

    if not SAMPLE.exists():
        print(f"Sample dataset missing; run: python scripts/make_sample_dataset.py")
        return 2

    step(1, "health of both services")
    ml = requests.get(f"{a.ml}/health", timeout=10).json()
    rc = requests.get(f"{a.receiver}/health", timeout=10).json()
    print("ML       :", ml["status"], "| entities:", ml["n_entities"])
    print("Receiver :", rc["status"], "| db:", rc["database"], "| events:",
          rc.get("events_stored"))

    step(2, "upload dataset + config (discovery is scoped to THIS file)")
    with open(SAMPLE, "rb") as fh:
        r = requests.post(f"{a.ml}/v1/datasets/upload", headers=H,
                          files={"file": (SAMPLE.name, fh, "text/csv")},
                          data={"config": json.dumps(CONFIG)}, timeout=300)
    r.raise_for_status()
    up = r.json()
    print("upload_id:", up["upload_id"], "| entities found:", up["n_entities"])
    for e in up["entities"]:
        print(f"  {e['entity_id']:8s} rows={e['records']:5d} days={e['days']:3d} "
              f"trainable={e['trainable']!s:5s} url={e['score_url']}")
        for reason in e["blocking_reasons"]:
            print(f"      blocked: {reason}")

    step(3, f"score {TARGET} BEFORE promotion (expect 409 MODEL_NOT_READY)")
    df = pd.read_csv(SAMPLE, parse_dates=["DVRDate", "next_DVRDate"])
    sub = df[(df.KDVR == 215) & (df.ChanNo == 0)].sort_values("DVRDate")
    recs = [{"DVRDate": x.DVRDate.isoformat(), "next_DVRDate": x.next_DVRDate.isoformat(),
             "Count": float(x.Count), "Count_IN": float(x.Count_IN),
             "Count_OUT": float(x.Count_OUT)} for x in sub.tail(60).head(20).itertuples()]
    pre = requests.post(f"{a.ml}/v1/sites/163/entities/215/0/score", headers=H,
                        json={"records": recs}, timeout=60)
    print("status:", pre.status_code, "->",
          (pre.json().get("detail") or {}).get("code", pre.text[:120]))

    step(4, f"train ONLY {TARGET} from this upload")
    r = requests.post(f"{a.ml}/v1/training/jobs", headers=H, timeout=60,
                      json={"upload_id": up["upload_id"], "entity_ids": [TARGET]})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print("job:", job_id)
    for note in r.json()["notes"]:
        print("  note:", note)
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        job = requests.get(f"{a.ml}/v1/training/jobs/{job_id}", headers=H,
                           timeout=30).json()
        if job["status"] in ("succeeded", "partial", "failed"):
            break
        print(f"  ... {job['status']} {job['progress']:.0%} {job.get('message','')}")
        time.sleep(5)
    print("job status:", job["status"])
    res = job["results"].get(TARGET, {})
    model_id = res.get("model_id")
    print("model_id:", model_id)

    step(5, "automatic calibration")
    cal = requests.get(f"{a.ml}/v1/training/jobs/{job_id}/calibration", headers=H,
                       timeout=30).json()
    print(f"status={cal['status']} scope={cal['scope']} method={cal['method']}")
    print(f"samples={cal['n_samples']} events={cal['n_positive']} "
          f"brier={cal['brier']} (uncalibrated {cal['brier_uncalibrated']}) "
          f"ece={cal['ece']} coverage={cal['coverage']}")

    step(6, "three backtest reports")
    bts = requests.get(f"{a.ml}/v1/training/jobs/{job_id}/backtests", headers=H,
                       timeout=30).json()
    for t, rep in bts["entities"][TARGET].items():
        print(f"  {t:24s} {rep['status']:16s} {rep.get('report_id','')}")
        print(f"      {rep.get('metrics_summary')}")

    step(7, "manual promotion (never automatic)")
    pr = requests.post(f"{a.ml}/v1/sites/163/entities/215/0/models/{model_id}/promote",
                       headers=H, json={"promoted_by": "smoke"}, timeout=60)
    print("status:", pr.status_code, "->", pr.json().get("model_status"))
    for other in ("215/1", "300/2"):
        rr = requests.get(f"{a.ml}/v1/entities/{other}/model", headers=H,
                          timeout=30).json()
        print(f"  {other} untouched: {rr['model_status']}")

    step(8, "score via the entity-specific URL")
    sc = requests.post(f"{a.ml}/v1/sites/163/entities/215/0/score", headers=H,
                       json={"records": recs}, timeout=120)
    sc.raise_for_status()
    b = sc.json()
    print(f"decision_time={b['decision_time']} cart={b['current_cart']} "
          f"risk={b['shortage_risk']} alert={b['alert_state']}")
    print(f"forecast 20m={b['forecast']['cart_at_20min']} "
          f"min40m={b['forecast']['min_within_40min']} "
          f"lower={b['forecast']['lower_bound_min_within_40min']}")
    print(f"action: replenish={b['action']['should_replenish']} "
          f"qty={b['action']['quantity']} urgency={b['action']['urgency']}")
    print("model:", b["model_id"], "| calibration scope:", b["calibration_scope"])
    print("CSV artifact:", b["artifact_path"])
    for w in b["warnings"]:
        print("  warning:", w)

    step(9, "idempotent retry reuses the artifact")
    again = requests.post(f"{a.ml}/v1/sites/163/entities/215/0/score", headers=H,
                          json={"records": recs}, timeout=120).json()
    print("replay:", again["idempotent_replay"],
          "| same artifact:", again["artifact_path"] == b["artifact_path"])

    step(10, "outbox delivery to the receiver")
    for _ in range(12):
        ob = requests.get(f"{a.ml}/v1/integrations/outbox", headers=H,
                          timeout=30).json()["counts"]
        if ob["pending"] == 0 and ob["sending"] == 0:
            break
        time.sleep(3)
    print("outbox:", ob)

    step(11, "query the receiver")
    for kind in ("training", "calibration", "backtest", "scoring"):
        n = requests.get(f"{a.receiver}/v1/results/{kind}",
                         params={"entity_id": TARGET}, timeout=30).json()["total"]
        print(f"  {kind:12s} {n}")
    bt = requests.get(f"{a.receiver}/v1/results/backtest",
                      params={"job_id": job_id}, timeout=30).json()
    print("  backtest types delivered:", sorted(i["backtest_type"] for i in bt["items"]))
    summ = requests.get(f"{a.receiver}/v1/results/entities/{TARGET}",
                        timeout=30).json()
    print("  entity summary:", summ["counts"])

    print(f"\n{BAR}\nSMOKE TEST COMPLETE\n{BAR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
