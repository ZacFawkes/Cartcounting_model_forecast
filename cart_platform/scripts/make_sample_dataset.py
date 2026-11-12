"""Generate a multi-entity sample dataset for the on-demand workflow.

Derives three (KDVR, ChanNo, BayID) streams from the real Site 163 telemetry so
the upload contains more than one bay and selected-entity training can be
demonstrated without touching the others.

    python scripts/make_sample_dataset.py [out.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cart_forecasting" / "data" / "real_data.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "sample_multi_entity.csv"

# Bay identity is part of the entity id, so the sample carries real bay keys.
BAYS = {
    (215, 0): "06053B44_0A47_42FC_BE3E_F88CB85F99D8",
    (215, 1): "1B1C2D3E_4F50_6172_8394_A5B6C7D8E9F0",
    (300, 2): "2C2D3E4F_5061_7283_94A5_B6C7D8E9F001",
}

# (kdvr, chan_no, site_key, cart_scale, traffic_scale, seed)
STREAMS = [
    (215, 0, 163, 1.00, 1.00, 1),
    (215, 1, 163, 1.45, 1.20, 2),
    (300, 2, 177, 0.70, 0.85, 3),
]


def build(source: Path = SOURCE) -> pd.DataFrame:
    base = pd.read_csv(source, sep="\t", encoding="utf-8-sig")
    base.columns = [c.strip().lstrip("\ufeff") for c in base.columns]
    base["DVRDate"] = pd.to_datetime(base["DVRDate"])
    base["next_DVRDate"] = pd.to_datetime(base["next_DVRDate"])
    base = base.dropna(subset=["DVRDate", "next_DVRDate"]).sort_values("DVRDate")

    frames = []
    for kdvr, chan, site, cart_scale, traffic_scale, seed in STREAMS:
        rng = np.random.default_rng(seed)
        df = base.copy()
        df["KDVR"] = kdvr
        df["ChanNo"] = chan
        df["siteKey"] = site
        df["BayID"] = BAYS[(kdvr, chan)]
        df["AreaID"] = 69 + chan
        noise = rng.normal(0, 1.5, len(df))
        df["Count"] = np.clip(
            np.round(df["Count"].to_numpy() * cart_scale + noise), 0, None
        ).astype(int)
        df["Count_IN"] = np.clip(
            np.round(df["Count_IN"].to_numpy() * traffic_scale), 0, None
        ).astype(int)
        df["Count_OUT"] = np.clip(
            np.round(df["Count_OUT"].to_numpy() * traffic_scale), 0, None
        ).astype(int)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # Interleave the streams so the file is not conveniently grouped - this is
    # what a real export looks like and it exercises entity isolation.
    out = out.sort_values(["DVRDate", "KDVR", "ChanNo"]).reset_index(drop=True)
    out["CountID"] = range(1, len(out) + 1)
    out["CNN_Index"] = range(1, len(out) + 1)
    cols = ["CountID", "DVRDate", "next_DVRDate", "KDVR", "ChanNo", "CatID",
            "AreaID", "Count", "CNN_Index", "siteKey", "BayID", "Count_IN",
            "Count_OUT"]
    return out[[c for c in cols if c in out.columns]]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(out_path, index=False)
    summary = (
        df.groupby(["KDVR", "ChanNo", "BayID"])
        .agg(rows=("Count", "size"), site=("siteKey", "first"),
             cart_min=("Count", "min"), cart_max=("Count", "max"))
    )
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
