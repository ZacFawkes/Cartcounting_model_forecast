"""Alert bands discovered by K-Means (K=3) on the calibrated shortage risk.

The operator should not have to invent ``risk_threshold``. After calibration the
trainer clusters this entity's own risk distribution into three groups and takes
the band edges from where the clusters actually separate:

    NORMAL  <  band_warning  <=  WARNING  <  band_critical  <=  CRITICAL

Fitted on the **calibration split** — the same rows the conformal calibrator was
fitted on. Those are open-loop predictions, so the risks are not inflated by the
simulated replenishment surplus that closed-loop scoring adds.

Honesty rules, consistent with calibration:
  * boundaries sit between clusters (max of the lower, min of the upper), so a
    band edge is always a value the data actually separated at;
  * clusters are ordered by centroid, never by the arbitrary label K-Means
    assigns, so the result is reproducible;
  * a distribution that cannot support three bands returns ``insufficient_data``
    and the system default stands, rather than a fabricated threshold.

The degenerate case is real and was measured: on a short window with no
high-risk records, K-Means happily returns bands at 0.006 / 0.042, which would
mark almost every record CRITICAL. The guards below exist for that.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from settings import (
    ALERT_BAND_MIN_EVENTS,
    ALERT_BAND_MIN_SAMPLES,
    ALERT_BAND_MIN_WARNING,
    ALERT_BAND_SEED,
    DEFAULT_BAND_CRITICAL,
    DEFAULT_BAND_WARNING,
    EVENT_RISK_FLOOR,
)

LOGGER = logging.getLogger("platform.alert_bands")

FITTED = "fitted"
INSUFFICIENT = "insufficient_data"
FAILED = "failed"

N_CLUSTERS = 3


def _fallback(status: str, reason: str, diagnostics: dict,
              warnings: list) -> dict:
    """No usable clustering: say so and hand back the system default."""
    return {
        "status": status,
        "reason": reason,
        "bands": {"warning": float(DEFAULT_BAND_WARNING),
                  "critical": float(DEFAULT_BAND_CRITICAL)},
        "source": "system_default",
        "clusters": [],
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def fit_bands(risk, seed: int = ALERT_BAND_SEED) -> dict:
    """Cluster 1-D risk into three bands. Pure function, no side effects."""
    r = np.asarray(risk, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    n_events = int((r >= EVENT_RISK_FLOOR).sum())
    n_distinct = int(np.unique(r).size)

    diagnostics = {
        "n_samples": n,
        "n_events": n_events,
        "n_distinct": n_distinct,
        "event_risk_floor": float(EVENT_RISK_FLOOR),
        "risk_mean": float(r.mean()) if n else float("nan"),
        "risk_max": float(r.max()) if n else float("nan"),
        "quantiles": {f"q{int(q * 100)}": float(np.quantile(r, q))
                      for q in (0.5, 0.75, 0.9, 0.95, 0.99)} if n else {},
    }

    # --- guards -----------------------------------------------------------
    if n < ALERT_BAND_MIN_SAMPLES:
        return _fallback(
            INSUFFICIENT,
            f"alert banding needs >= {ALERT_BAND_MIN_SAMPLES} calibration "
            f"records; this entity has {n}",
            diagnostics, [])
    if n_distinct < N_CLUSTERS:
        # A flat distribution (e.g. every risk 0.0) has no three groups to find.
        # KMeans would raise on the empty-cluster reduction.
        return _fallback(
            INSUFFICIENT,
            f"risk takes only {n_distinct} distinct value(s); three bands "
            f"cannot be separated",
            diagnostics, [])
    if n_events < ALERT_BAND_MIN_EVENTS:
        return _fallback(
            INSUFFICIENT,
            f"only {n_events} record(s) with risk >= {EVENT_RISK_FLOOR}; "
            f">= {ALERT_BAND_MIN_EVENTS} are needed before a high band is "
            f"meaningful",
            diagnostics, [])

    # --- cluster ----------------------------------------------------------
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=N_CLUSTERS, n_init=10,
                    random_state=seed).fit(r.reshape(-1, 1))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("K-Means failed")
        return _fallback(FAILED, f"clustering failed: {exc}", diagnostics,
                         [f"alert banding failed: {exc}"])

    centres = km.cluster_centers_.ravel()
    order = np.argsort(centres)                       # ascending, deterministic
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[int(x)] for x in km.labels_])

    sizes = [int((labels == i).sum()) for i in range(N_CLUSTERS)]
    if min(sizes) == 0:  # pragma: no cover - sklearn avoids this
        return _fallback(INSUFFICIENT, "clustering produced an empty group",
                         {**diagnostics, "cluster_sizes": sizes}, [])

    clusters = []
    for i, name in enumerate(("NORMAL", "WARNING", "CRITICAL")):
        part = r[labels == i]
        clusters.append({
            "label": name, "n": int(part.size),
            "share": float(part.size / n),
            "risk_min": float(part.min()), "risk_max": float(part.max()),
            "centre": float(centres[order[i]]),
        })

    # Boundary sits between the clusters, not at a centroid: the midpoint of
    # the gap the data itself left.
    
    # band_w = clusters[1]["risk_min"]
    # band_c = clusters[2]["risk_min"]
    band_w = (clusters[0]["risk_max"] + clusters[1]["risk_min"]) / 2.0
    band_c = (clusters[1]["risk_max"] + clusters[2]["risk_min"]) / 2.0
    diagnostics["cluster_sizes"] = sizes
    diagnostics["inertia"] = float(km.inertia_)

    warnings: list[str] = []
    if band_w < ALERT_BAND_MIN_WARNING:
        return _fallback(
            INSUFFICIENT,
            f"clustered WARNING band is {band_w:.4f}, below the {ALERT_BAND_MIN_WARNING} "
            f"floor; this risk distribution is too flat to band and almost every "
            f"record would be elevated",
            {**diagnostics, "rejected_bands": {"warning": float(band_w),
                                               "critical": float(band_c)}},
            [])
    if not band_c > band_w:  # pragma: no cover - ordering makes this unreachable
        return _fallback(INSUFFICIENT,
                         f"bands are not ordered ({band_w:.4f} >= {band_c:.4f})",
                         diagnostics, [])

    if clusters[2]["share"] < 0.005:
        warnings.append(
            f"the CRITICAL cluster holds {clusters[2]['n']} record(s) "
            f"({clusters[2]['share']:.2%}); the upper band is estimated from very "
            f"few observations")
    if clusters[0]["share"] > 0.97:
        warnings.append(
            f"{clusters[0]['share']:.1%} of calibration records fall in NORMAL; "
            f"the bands are driven by a small tail")

    LOGGER.info("alert bands fitted: WARNING>=%.4f CRITICAL>=%.4f "
                "(sizes %s, centres %.4f/%.4f/%.4f)",
                band_w, band_c, sizes, *[c["centre"] for c in clusters])

    return {
        "status": FITTED,
        "reason": None,
        "bands": {"warning": float(band_w), "critical": float(band_c)},
        "source": "model_kmeans",
        "clusters": clusters,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def auto_band(
    entity_id: str,
    model_id: str,
    job_id: str,
    risk,
    artifact_dir: Path,
    config_hash: str | None = None,
    seed: int = ALERT_BAND_SEED,
) -> dict:
    """Fit, write the cluster table for audit, and return the record."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    band_id = f"bnd_{entity_id.replace(':', '_')}_{uuid.uuid4().hex[:8]}"

    res = fit_bands(risk, seed=seed)
    if res["clusters"]:
        pd.DataFrame(res["clusters"]).to_csv(
            artifact_dir / "risk_clusters.csv", index=False)

    if res["status"] != FITTED:
        LOGGER.warning("%s: alert bands not fitted — %s", entity_id, res["reason"])

    return {
        "band_id": band_id,
        "entity_id": entity_id,
        "model_id": model_id,
        "job_id": job_id,
        "config_hash": config_hash,
        "method": f"kmeans_k{N_CLUSTERS}",
        "seed": int(seed),
        **res,
        "artifact_dir": str(artifact_dir),
    }


def bands_from_meta(meta: dict | None) -> dict:
    """Usable bands from a model row, or {} when there are none.

    Only a fitted result is honoured. ``insufficient_data`` and ``failed``
    deliberately return nothing so the documented default stands rather than a
    fabricated operating point.
    """
    if not meta or meta.get("alert_band_status") != FITTED:
        return {}
    from registry.database import from_json

    record = from_json(meta.get("alert_band_json")) or {}
    bands = record.get("bands") or {}
    w, c = bands.get("warning"), bands.get("critical")
    if w is None or c is None:
        return {}
    return {"warning": float(w), "critical": float(c)}
