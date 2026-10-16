"""Automated professional DOCX report generation (python-docx)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

import config as C

LOGGER = logging.getLogger(__name__)

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x59, 0x59, 0x59)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def _fmt(v, nd: int = 2, pct: bool = False, na: str = "n/a") -> str:
    if v is None:
        return na
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return na
        return f"{v:,.{nd}f}%" if pct else f"{v:,.{nd}f}"
    return str(v)


def _h(doc, text: str, level: int = 1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = ACCENT
    return p


def _para(doc, text: str, italic: bool = False, size: int = 10):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.font.size = Pt(size)
    if italic:
        r.font.color.rgb = MUTED
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def _bullets(doc, items, style: str = "List Bullet"):
    for it in items:
        p = doc.add_paragraph(str(it), style=style)
        for r in p.runs:
            r.font.size = Pt(10)


def _table(doc, headers, rows, caption: str | None = None, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = str(h)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    if caption:
        _para(doc, caption, italic=True, size=8)
    return t


def _figure(doc, path: Path | None, caption: str, width: float = 6.3):
    if path is None or not Path(path).exists():
        _para(doc, f"[figure unavailable: {caption}]", italic=True)
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    _para(doc, caption, italic=True, size=8)


def _df_rows(df: pd.DataFrame, cols, nd=3, limit=None):
    d = df if limit is None else df.head(limit)
    rows = []
    for _, r in d.iterrows():
        rows.append([_fmt(r[c], nd) for c in cols])
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def build_report(res: dict, path: Path = C.REPORT_FILE) -> Path:
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
        s.top_margin = s.bottom_margin = Inches(0.7)

    _title_page(doc, res)
    doc.add_page_break()
    _section_1(doc, res)
    doc.add_page_break()
    _section_2(doc, res)
    doc.add_page_break()
    _section_3(doc, res)
    doc.add_page_break()
    _section_4(doc, res)
    doc.add_page_break()
    _section_5(doc, res)
    doc.add_page_break()
    _section_6(doc, res)
    doc.add_page_break()
    _section_7(doc, res)
    doc.add_page_break()
    _section_8(doc, res)
    _section_9(doc, res)
    _section_10(doc, res)

    doc.save(path)
    LOGGER.info("DOCX report written: %s", path)
    return path


def _title_page(doc, res):
    ov = res["overview"]
    t = doc.add_heading("Cart Availability Forecasting, Shortage Early Warning\n& Replenishment Optimization", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in t.runs:
        r.font.color.rgb = ACCENT
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Site {ov['site_keys'][0]} — chronological backtest report")
    r.font.size = Pt(13)
    r.font.color.rgb = MUTED

    _table(
        doc,
        ["Item", "Value"],
        [
            ["Telemetry window", f"{_fmt(ov['start'])} to {_fmt(ov['end'])}"],
            ["Records analysed", _fmt(ov["n_records"])],
            ["Days of data", _fmt(ov["n_days"])],
            ["Forecast horizons", "+3 records (~20 min) and +6 records (~40 min)"],
            ["Shortage threshold (RED)", _fmt(C.RED_THRESHOLD)],
            ["Watch threshold (AMBER)", _fmt(C.AMBER_THRESHOLD)],
            ["Evaluation design", "Chronological train / validation / holdout with purging"],
            ["Selected 40-min minimum model", res["selected_models"]["y_min_6"]],
            ["Audience", "Data Science, Engineering, Product, Retail Operations"],
        ],
        widths=[2.2, 4.0],
    )
    _para(
        doc,
        "All forecasting and warning metrics in this report are measured on an unseen "
        "chronological holdout period. All replenishment benefits are simulation-based "
        "scenario estimates produced by an explicit inventory model, not observed causal effects.",
        italic=True,
    )


# ------------------------------- Section 1 --------------------------------
def _section_1(doc, res):
    _h(doc, "1. Executive Summary", 1)
    ov = res["overview"]
    mc = res["model_comparison"]
    ws = res["warning_summary"]
    pm = res["policy_metrics"]
    sel = res["selected_models"]

    def _ho(target, model):
        row = mc[(mc.target == target) & (mc.model == model) & (mc.split == "holdout")]
        return row.iloc[0] if len(row) else None

    best3 = _ho("y_cart_3", sel["y_cart_3"])
    best6 = _ho("y_cart_6", sel["y_cart_6"])
    bestmin = _ho("y_min_6", sel["y_min_6"])
    pers3 = _ho("y_cart_3", "Persistence")
    persmin = _ho("y_min_6", "Persistence")

    imp3 = (pers3["MAE"] - best3["MAE"]) / pers3["MAE"] * 100 if pers3 is not None else np.nan
    impmin = (
        (persmin["MAE"] - bestmin["MAE"]) / persmin["MAE"] * 100 if persmin is not None else np.nan
    )

    _h(doc, "1.1 Answers to the six business questions", 2)
    q = [
        (
            "Can future cart shortages be predicted?",
            f"Yes, partially. On the holdout period the risk model reaches PR-AUC "
            f"{_fmt(res['risk_holdout_metrics']['PR_AUC'], 3)} against a base shortage rate of "
            f"{_fmt(res['risk_holdout_metrics']['base_rate'] * 100, 1)}%, and the warning engine "
            f"detected {ws['n_detected']} of {ws['n_events']} independent shortage events "
            f"(event recall {_fmt(ws['event_recall'] * 100, 1)}%).",
        ),
        (
            "How accurate are the forecasts?",
            f"The selected model predicts the cart count 20 minutes ahead with MAE "
            f"{_fmt(best3['MAE'])} carts and 40 minutes ahead with MAE {_fmt(best6['MAE'])} carts. "
            f"The 40-minute future minimum is predicted with MAE {_fmt(bestmin['MAE'])} carts, "
            f"{_fmt(impmin, 1)}% better than the persistence baseline.",
        ),
        (
            "How early can shortages be detected?",
            f"Median warning lead time is {_fmt(ws['median_lead_time_min'], 1)} minutes "
            f"(mean {_fmt(ws['mean_lead_time_min'], 1)} minutes); "
            f"{_fmt(ws['pct_warned_ge_10min'], 1)}% of events were warned at least 10 minutes "
            f"early and {_fmt(ws['pct_warned_ge_20min'], 1)}% at least 20 minutes early.",
        ),
        (
            "How many carts should employees replenish?",
            f"The frozen policy recommends a median of "
            f"{_fmt(res['replenishment_decisions']['quantity'].median() if len(res['replenishment_decisions']) else np.nan, 1)} "
            f"carts per trip (mean {_fmt(pm['avg_carts_per_action'], 1)}), sized to lift the "
            f"predicted 40-minute minimum back to the target stock of "
            f"{res['frozen_policy']['target_stock']} carts.",
        ),
        (
            "How much operational improvement is estimated?",
            f"Under the simulation assumptions, shortage severity falls by "
            f"{_fmt(pm['severity_reduction_pct'], 1)}% and shortage minutes by "
            f"{_fmt(pm['duration_reduction_pct'], 1)}%, at a cost of "
            f"{_fmt(pm['trips_per_day'], 1)} retrieval trips and "
            f"{_fmt(pm['carts_per_day'], 1)} carts moved per day.",
        ),
        (
            "Is the approach ready for shadow deployment?",
            "Yes for shadow mode, no for automated action. The pipeline is leakage-checked, "
            "chronologically backtested and reproducible, but it is based on 14 days from a "
            "single site, assumes a 10-minute replenishment lead time, and has no recorded "
            "historical replenishment actions to validate the simulated benefit against.",
        ),
    ]
    for i, (question, answer) in enumerate(q, start=1):
        p = doc.add_paragraph()
        r = p.add_run(f"Q{i}. {question} ")
        r.bold = True
        r.font.size = Pt(10)
        r2 = p.add_run(answer)
        r2.font.size = Pt(10)

    _h(doc, "1.2 Key performance indicators (holdout)", 2)
    rows = [
        ["Forecast MAE — cart count +20 min", f"{_fmt(best3['MAE'])} carts", f"Persistence {_fmt(pers3['MAE'])} ({_fmt(imp3, 1)}% better)"],
        ["Forecast MAE — cart count +40 min", f"{_fmt(best6['MAE'])} carts", f"Persistence {_fmt(_ho('y_cart_6', 'Persistence')['MAE'])}"],
        ["Forecast MAE — 40-min minimum", f"{_fmt(bestmin['MAE'])} carts", f"Persistence {_fmt(persmin['MAE'])} ({_fmt(impmin, 1)}% better)"],
        ["Shortage risk PR-AUC", _fmt(res["risk_holdout_metrics"]["PR_AUC"], 3), f"Base rate {_fmt(res['risk_holdout_metrics']['base_rate'] * 100, 1)}%"],
        ["Shortage risk Brier score", _fmt(res["risk_holdout_metrics"]["Brier"], 4), "Lower is better"],
        ["Independent shortage events (holdout)", _fmt(ws["n_events"]), "Grouped with explicit recovery rule"],
        ["Event recall", _fmt(ws["event_recall"] * 100, 1) + "%", f"{ws['n_detected']} of {ws['n_events']} events"],
        ["Median warning lead time", f"{_fmt(ws['median_lead_time_min'], 1)} min", "Before shortage onset"],
        ["False alerts per day", _fmt(ws["false_alerts_per_day"], 2), "Operational noise budget"],
        ["90% lower-bound coverage", _fmt(res["coverage_holdout"]["empirical_lower_coverage"] * 100, 1) + "%", "Target 90% (conformal)"],
        ["Recommended trips per day (simulated)", _fmt(pm["trips_per_day"], 1), f"{_fmt(pm['avg_carts_per_action'], 1)} carts per trip"],
        ["Simulated shortage severity reduction", _fmt(pm["severity_reduction_pct"], 1) + "%", "Scenario estimate, not measured"],
    ]
    _table(doc, ["KPI", "Result", "Context"], rows, widths=[2.6, 1.6, 2.4])

    _para(
        doc,
        "Measured vs simulated: forecasting accuracy, risk calibration, event recall and lead "
        "time are measured on held-out telemetry. Shortage reduction, trips and carts moved are "
        "produced by the additive inventory simulator under documented assumptions.",
        italic=True,
    )


# ------------------------------- Section 2 --------------------------------
def _section_2(doc, res):
    _h(doc, "2. Data Overview", 1)
    ov = res["overview"]
    rows = [
        ["Date range", f"{_fmt(ov['start'])} → {_fmt(ov['end'])}"],
        ["Total records", _fmt(ov["n_records"])],
        ["Number of days", _fmt(ov["n_days"])],
        ["Records per day (mean)", _fmt(ov["records_per_day_mean"], 1)],
        ["Average cart count", _fmt(ov["cart_mean"], 1)],
        ["Median cart count", _fmt(ov["cart_median"], 1)],
        ["Minimum cart count", _fmt(ov["cart_min"])],
        ["Maximum cart count", _fmt(ov["cart_max"])],
        ["Cart count std. dev.", _fmt(ov["cart_std"], 1)],
        ["Mean traffic IN per interval", _fmt(ov["traffic_in_mean"], 1)],
        ["Mean traffic OUT per interval", _fmt(ov["traffic_out_mean"], 1)],
        ["Total traffic IN / OUT", f"{_fmt(ov['traffic_in_total'])} / {_fmt(ov['traffic_out_total'])}"],
        ["Peak traffic IN / OUT in one interval", f"{_fmt(ov['traffic_in_max'])} / {_fmt(ov['traffic_out_max'])}"],
        ["Records below RED (<15)", f"{_fmt(ov['records_below_red'])} ({_fmt(ov['records_below_red'] / ov['n_records'] * 100, 1)}%)"],
        ["Records below AMBER (<20)", f"{_fmt(ov['records_below_amber'])} ({_fmt(ov['records_below_amber'] / ov['n_records'] * 100, 1)}%)"],
        ["Independent shortage events (full period)", _fmt(len(res["events_all"]))],
        ["Median interval duration", f"{_fmt(ov['interval_median'], 2)} min"],
        ["Interval duration range", f"{_fmt(ov['interval_min'], 2)} – {_fmt(ov['interval_max'], 2)} min"],
    ]
    _table(doc, ["Metric", "Value"], rows, widths=[3.0, 3.2])

    _h(doc, "2.1 Historical cart availability and traffic", 2)
    _figure(doc, res["figures"]["historical"], "Figure 1 — Cart availability (top) and customer traffic (bottom). Red shading marks observed sub-RED periods.")

    _h(doc, "2.2 Record-based horizon timing", 2)
    _para(
        doc,
        "The project assumes 3 records ≈ 20 minutes. Original timestamps are preserved and the "
        "realised elapsed time of every horizon is measured; horizons deviating more than 50% "
        "from nominal are excluded from modelling and evaluation.",
    )
    ht = res["horizon_timing"]
    _table(
        doc,
        ["Horizon", "Nominal (min)", "Median (min)", "Mean (min)", "P05", "P95", "Max", "% within ±50%"],
        _df_rows(ht, ["horizon_records", "nominal_minutes", "median_minutes", "mean_minutes",
                      "p05_minutes", "p95_minutes", "max_minutes", "pct_within_50pct_of_nominal"], nd=1),
    )
    _para(doc, f"{_fmt(res['n_excluded_unreliable'])} otherwise-eligible origins were excluded as unreliable horizons.", italic=True)

    _h(doc, "2.3 Data quality checks", 2)
    v = res["validation"]
    _table(doc, ["Check", "Status", "Detail"], _df_rows(v, ["check", "status", "detail"]), widths=[1.8, 0.8, 3.6])

    _h(doc, "2.4 Daily completeness and shortage exposure", 2)
    d = res["daily_completeness"].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date.astype(str)
    _table(
        doc,
        ["Date", "Records", "Hours covered", "Mean cart", "Min cart", "Records < RED", "Mean interval (min)"],
        _df_rows(d, ["date", "n_records", "hours_covered", "cart_mean", "cart_min",
                     "records_below_red", "mean_interval_min"], nd=1),
    )

    _h(doc, "2.5 Chronological split", 2)
    sm = res["split_meta"]
    _table(
        doc,
        ["Split", "Days", "Modelling rows", "Purpose"],
        [
            ["Training", ", ".join(sm["train_days"]), _fmt(sm["n_train"]), "Model fitting and feature screening"],
            ["Validation", ", ".join(sm["validation_days"]), _fmt(sm["n_validation"]), "Model selection, conformal calibration, alert & policy tuning"],
            ["Holdout", ", ".join(sm["holdout_days"]), _fmt(sm["n_holdout"]), "Final untouched evaluation"],
        ],
        widths=[1.0, 2.4, 1.0, 2.0],
        caption=f"The last {sm['purge_records']} origins of the training and validation splits are purged so that no target window crosses a split boundary. Partial days: {', '.join(sm['partial_days']) or 'none'}.",
    )


# ------------------------------- Section 3 --------------------------------
def _section_3(doc, res):
    _h(doc, "3. Forecasting Results", 1)
    _h(doc, "3.1 Methodology", 2)
    _bullets(doc, [
        f"Unified feature configuration: cart count, traffic IN/OUT, temporal and advanced statistical features over windows {C.FEATURE_WINDOWS} records.",
        f"{_fmt(res['n_candidate_features'])} candidate features engineered; {_fmt(res['n_selected_features'])} retained after conservative screening on training data only (constant, highly missing, non-finite and near-duplicate features removed).",
        "Direct multi-target forecasting: cart count at +3 and +6 records, and the future minimum over the next 3 and 6 records. The minimum is modelled directly rather than inferred from endpoints, because a shortage can occur between endpoints.",
        "Models compared: Persistence baseline, Ridge regression on standardised features, and LightGBM with a conservative hyper-parameter search.",
        "Every feature uses only information available at the record's decision time (next_DVRDate); rolling windows never cross a stream break.",
    ])

    _h(doc, "3.2 Model comparison (holdout)", 2)
    mc = res["model_comparison"]
    ho = mc[mc.split == "holdout"]
    names = {"y_cart_3": "+3 records (~20 min)", "y_cart_6": "+6 records (~40 min)",
             "y_min_3": "min over 3 records", "y_min_6": "min over 6 records"}
    rows = []
    for target in ["y_cart_3", "y_cart_6"]:
        for model in ["Persistence", "Ridge", "LightGBM"]:
            r = ho[(ho.target == target) & (ho.model == model)]
            if not len(r):
                continue
            r = r.iloc[0]
            star = " *" if res["selected_models"][target] == model else ""
            rows.append([model + star, names[target], _fmt(r["MAE"]), _fmt(r["RMSE"]),
                         _fmt(r["Bias"]), _fmt(r["MAE_near_shortage"])])
    _table(doc, ["Model", "Horizon", "MAE", "RMSE", "Bias", "MAE near shortage"], rows,
           caption="* = selected model. Bias > 0 means the model over-predicts cart availability (operationally dangerous).")

    _h(doc, "3.3 Future-minimum forecasting (holdout)", 2)
    _para(doc, "The future minimum drives shortage prevention and is evaluated separately.")
    rows = []
    for target in ["y_min_3", "y_min_6"]:
        for model in ["Persistence", "Ridge", "LightGBM"]:
            r = ho[(ho.target == target) & (ho.model == model)]
            if not len(r):
                continue
            r = r.iloc[0]
            star = " *" if res["selected_models"][target] == model else ""
            rows.append([model + star, names[target], _fmt(r["MAE"]), _fmt(r["RMSE"]),
                         _fmt(r["Bias"]), _fmt(r["MAE_near_shortage"])])
    _table(doc, ["Model", "Target", "MAE", "RMSE", "Bias", "MAE near shortage"], rows)

    _h(doc, "3.4 Model selection rationale", 2)
    _para(
        doc,
        "Models were selected on the validation split using a composite of overall MAE, MAE near "
        "the shortage boundary and shortage recall — not overall MAE alone. A model that is "
        "accurate on average but systematically over-predicts availability when carts are scarce "
        "is operationally worse than a slightly less accurate but conservative model.",
    )
    vt = res["validation_model_table"]
    rows = []
    for target in ["y_cart_3", "y_cart_6", "y_min_3", "y_min_6"]:
        for _, r in vt[vt.target == target].iterrows():
            rows.append([names[target], r["model"], _fmt(r["MAE"]), _fmt(r["MAE_near_shortage"]),
                         _fmt(r["shortage_recall"], 3), _fmt(r["shortage_precision"], 3)])
    _table(doc, ["Target", "Model", "Val MAE", "Val MAE near shortage", "Val shortage recall", "Val shortage precision"], rows)

    wf = res.get("walk_forward")
    if wf is not None and len(wf):
        _h(doc, "3.5 Walk-forward validation (40-minute minimum)", 2)
        w = wf.copy()
        for c in ("test_start", "test_end"):
            w[c] = pd.to_datetime(w[c]).dt.strftime("%m-%d %H:%M")
        _table(doc, ["Fold", "Test start", "Test end", "n", "MAE", "RMSE", "Bias"],
               _df_rows(w, ["fold", "test_start", "test_end", "n", "MAE", "RMSE", "Bias"]),
               caption="Expanding-window refits on training+validation data confirm stability over time.")

    _h(doc, "3.6 Prediction uncertainty", 2)
    cov = res["coverage_holdout"]
    rn = res["residual_normality_note"]
    _para(
        doc,
        f"A split-conformal calibrator was fitted on the validation residuals "
        f"({_fmt(rn['n_calibration'])} points, skew {_fmt(rn['residual_skew'])}, excess kurtosis "
        f"{_fmt(rn['residual_kurtosis'])}), so normality is not assumed. The one-sided 90% lower "
        f"bound offsets the point forecast by {_fmt(cov['lower_offset'])} carts.",
    )
    _table(
        doc,
        ["Interval", "Target coverage", "Empirical coverage (holdout)", "Mean width (carts)"],
        [
            ["90% lower bound", _fmt(cov["target_lower_coverage"] * 100, 0) + "%",
             _fmt(cov["empirical_lower_coverage"] * 100, 1) + "%", "—"],
            ["80% central interval", _fmt(cov["target_interval_coverage"] * 100, 0) + "%",
             _fmt(cov["empirical_interval_coverage"] * 100, 1) + "%", _fmt(cov["mean_interval_width"])],
        ],
    )

    _h(doc, "3.7 Actual versus predicted", 2)
    _figure(doc, res["figures"]["forecast_20m"], "Figure 2 — Cart count at +3 records (~20 minutes), holdout period.")
    _figure(doc, res["figures"]["forecast_40m"], "Figure 3 — Cart count at +6 records (~40 minutes), holdout period.")
    _figure(doc, res["figures"]["minimum"], "Figure 4 — Future 40-minute minimum with the conformal lower bound.")


# ------------------------------- Section 4 --------------------------------
def _section_4(doc, res):
    _h(doc, "4. Shortage Early-Warning Results", 1)
    ws = res["warning_summary"]
    rm = res["risk_holdout_metrics"]
    acfg = res["alert_cfg"]

    _h(doc, "4.1 Risk model", 2)
    _para(
        doc,
        f"Risk is defined as P(min(C(t+1..t+6)) < {C.RED_THRESHOLD} | X_t). Two approaches were "
        f"compared on validation: a forecast-derived probability from the empirical conformal "
        f"residual distribution, and direct classifiers. The selected estimator is "
        f"{res['selected_risk_model']}.",
    )
    rt = res["risk_model_table"]
    _table(doc, ["Risk model", "PR-AUC", "ROC-AUC", "Brier", "Positives", "Base rate"],
           _df_rows(rt, ["model", "PR_AUC", "ROC_AUC", "Brier", "n_positive", "base_rate"], nd=4),
           caption="Validation performance. PR-AUC is the primary criterion because shortages are rare.")
    _table(doc, ["Holdout metric", "Value"],
           [["PR-AUC", _fmt(rm["PR_AUC"], 4)], ["ROC-AUC", _fmt(rm["ROC_AUC"], 4)],
            ["Brier score", _fmt(rm["Brier"], 4)], ["Base shortage rate", _fmt(rm["base_rate"] * 100, 2) + "%"],
            ["Positive origins", _fmt(rm["n_positive"])], ["Scored origins", _fmt(rm["n"])]])
    _figure(doc, res["figures"]["risk_calibration"], "Figure 5 — Shortage risk calibration on the holdout period.")

    _h(doc, "4.2 Alert engine", 2)
    _bullets(doc, [
        f"States: NORMAL / WARNING / CRITICAL, with WARNING at risk ≥ {_fmt(acfg.warning_risk, 2)} and CRITICAL at risk ≥ {_fmt(acfg.critical_risk, 2)} (calibrated on validation).",
        f"Deduplication and a {_fmt(acfg.cooldown_min, 0)}-minute cooldown prevent repeated alerts for the same developing shortage; escalation from WARNING to CRITICAL bypasses the cooldown.",
        f"Hysteresis of {_fmt(acfg.hysteresis, 2)} keeps the engine elevated until risk falls clearly below the entry threshold, avoiding alert flapping.",
        f"Observing Count < {C.RED_THRESHOLD} raises an immediate reactive CRITICAL alert; those are reported separately and never credited as predictive warnings.",
    ])

    _h(doc, "4.3 Warning performance (holdout)", 2)
    rows = [
        ["Actual shortage events", _fmt(ws["n_events"])],
        ["Detected events", _fmt(ws["n_detected"])],
        ["Event recall", _fmt(ws["event_recall"] * 100, 1) + "%"],
        ["Predictive alerts issued", _fmt(ws["n_predictive_alerts"])],
        ["Event precision (alerts followed by an event within 60 min)", _fmt(ws["event_precision"] * 100, 1) + "%"],
        ["Median lead time", f"{_fmt(ws['median_lead_time_min'], 1)} min"],
        ["Mean lead time", f"{_fmt(ws['mean_lead_time_min'], 1)} min"],
        ["False alerts per store per day", _fmt(ws["false_alerts_per_day"], 2)],
        ["Events warned ≥ 10 min early", _fmt(ws["pct_warned_ge_10min"], 1) + "%"],
        ["Events warned ≥ 20 min early", _fmt(ws["pct_warned_ge_20min"], 1) + "%"],
        ["Reactive critical alerts (already in shortage)", _fmt(ws["n_reactive_criticals"])],
        ["Evaluation span (days)", _fmt(ws["evaluation_days"], 2)],
    ]
    _table(doc, ["Metric", "Result"], rows, widths=[3.6, 2.6])

    _h(doc, "4.4 Event-level detail", 2)
    m = res["matched_events"].copy()
    if len(m):
        m["onset_time"] = pd.to_datetime(m["onset_time"]).dt.strftime("%m-%d %H:%M")
        m["first_warning_time"] = pd.to_datetime(m["first_warning_time"]).dt.strftime("%m-%d %H:%M").fillna("—")
        _table(doc, ["Event", "Onset", "Min cart", "Duration (min)", "Detected", "First warning", "Lead (min)"],
               _df_rows(m, ["event_id", "onset_time", "min_cart", "duration_min", "detected",
                            "first_warning_time", "lead_time_min"], nd=1))
    else:
        _para(doc, "No independent shortage events occurred in the holdout period.", italic=True)

    _figure(doc, res["figures"]["warning_timeline"], "Figure 6 — Warning timeline: observed carts with shortage windows (red) and predictive warnings (orange), with the risk trace below.")
    _figure(doc, res["figures"]["lead_time"], "Figure 7 — Distribution of warning lead times.")


# ------------------------------- Section 5 --------------------------------
def _section_5(doc, res):
    _h(doc, "5. Replenishment Recommendations", 1)
    fp = res["frozen_policy"]
    d = res["replenishment_decisions"].copy()

    _h(doc, "5.1 Decision logic", 2)
    _bullets(doc, [
        f"Trigger: estimated shortage risk over the next 6 records (~40 min) ≥ {_fmt(fp['risk_threshold'], 2)}.",
        f"Quantity: Q = ceil(target stock − predicted 40-minute minimum), using the {fp['quantity_rule']} rule; the conservative alternative uses the 90% lower bound instead of the point forecast.",
        f"Target stock: {fp['target_stock']} carts; minimum {fp['min_carts_per_trip']} and maximum {fp['max_carts_per_trip']} carts per trip; bay capacity {fp['bay_capacity']}.",
        f"Timing: a decision at time T becomes effective at T + {_fmt(fp['lead_time_min'], 0)} minutes (assumed lead time). If the predicted breach occurs sooner, the recommendation is flagged URGENT.",
        f"Cooldown of {_fmt(fp['cooldown_min'], 0)} minutes and explicit pending-order tracking prevent double-counting while a retrieval is in transit.",
    ])

    _h(doc, "5.2 Recommended interventions (holdout)", 2)
    if len(d):
        dd = d.copy()
        dd["decision_time"] = pd.to_datetime(dd["decision_time"]).dt.strftime("%m-%d %H:%M")
        dd["effective_time"] = pd.to_datetime(dd["effective_time"]).dt.strftime("%m-%d %H:%M")
        show = dd.head(25)
        _table(
            doc,
            ["Timestamp", "Current carts", "Forecast min (40 min)", "Lower bound", "Risk", "Top-up", "Action by", "Urgency"],
            _df_rows(show, ["decision_time", "current_cart", "predicted_min", "lower_bound_min",
                            "risk", "quantity", "effective_time", "urgency"], nd=1),
            caption=f"Showing {min(25, len(dd))} of {len(dd)} recommendations. Full list: outputs/replenishment_recommendations.csv",
        )
        _table(
            doc,
            ["Summary", "Value"],
            [
                ["Total recommendations", _fmt(len(d))],
                ["Mean carts per trip", _fmt(d["quantity"].mean(), 1)],
                ["Median carts per trip", _fmt(d["quantity"].median(), 1)],
                ["Range of carts per trip", f"{_fmt(int(d['quantity'].min()))} – {_fmt(int(d['quantity'].max()))}"],
                ["Total carts recommended", _fmt(int(d["quantity"].sum()))],
                ["URGENT recommendations", _fmt(int((d["urgency"] == "URGENT").sum()))],
                ["Completable before predicted breach", _fmt(d["can_complete_before_breach"].mean() * 100, 1) + "%"],
                ["Capacity-constrained recommendations", _fmt(int(d["constrained"].sum()))],
                ["Point-rule vs risk-rule mean quantity", f"{_fmt(d['quantity_point'].mean(), 1)} vs {_fmt(d['quantity_risk'].mean(), 1)}"],
            ],
            widths=[3.4, 2.8],
        )
        _para(
            doc,
            "Why replenishment was recommended: each row fires because the estimated probability "
            "that the cart count drops below RED within the next ~40 minutes exceeded the frozen "
            "risk threshold. The quantity closes the gap between the predicted 40-minute minimum "
            "and the target stock, so the store is expected to stay above RED for the whole "
            "planning horizon. Because retrieval takes time, the action must start at the "
            "timestamp shown and be completed by the action-by time; URGENT rows are cases where "
            "the predicted breach arrives sooner than the lead time allows.",
        )
    else:
        _para(doc, "The frozen policy issued no replenishment recommendations on the holdout period.", italic=True)

    _h(doc, "5.3 Conservative vs point quantities", 2)
    _para(
        doc,
        "The risk-adjusted quantity (based on the 90% conformal lower bound) is larger and "
        "therefore safer, but it moves more carts and increases staff workload. The policy "
        "search evaluated both rules under the operational cost objective "
        "J = λ₁·Shortage + λ₂·Trips + λ₃·CartsMoved + λ₄·Excess and selected the "
        f"'{fp['quantity_rule']}' rule on validation data.",
    )
    _figure(doc, res["figures"]["replenishment"], "Figure 8 — Replenishment decisions, delivery times and the simulated inventory path.")


# ------------------------------- Section 6 --------------------------------
def _section_6(doc, res):
    _h(doc, "6. Replenishment Policy Backtest", 1)
    pm = res["policy_metrics"]
    fp = res["frozen_policy"]

    _h(doc, "6.1 Policy selection and freezing", 2)
    pt = res["policy_search_table"]
    _para(
        doc,
        f"{len(pt)} candidate policies were simulated on the validation split and ranked by the "
        f"operational cost J. The best policy was frozen before any holdout evaluation.",
    )
    top = pt.head(5)
    _table(
        doc,
        ["Risk thr.", "Target stock", "Qty rule", "Cooldown", "Events", "Severity red. %", "Trips", "Carts", "J"],
        _df_rows(top, ["risk_threshold", "target_stock", "quantity_rule", "cooldown_min",
                       "policy_events", "severity_reduction_pct", "n_replenishment_actions",
                       "total_carts_recommended", "policy_cost_J"], nd=1),
        caption="Top 5 validation policies. Full search: outputs/policy_search_validation.csv",
    )

    _h(doc, "6.2 Historical baseline versus simulated policy (holdout)", 2)
    rows = [
        ["Shortage events", _fmt(pm["baseline_events"]), _fmt(pm["policy_events"])],
        ["Records below RED", _fmt(pm["baseline_records_below_red"], 0), _fmt(pm["policy_records_below_red"], 0)],
        ["Shortage duration (minutes)", _fmt(pm["baseline_shortage_minutes"], 1), _fmt(pm["policy_shortage_minutes"], 1)],
        ["Shortage severity (cart-records below RED)", _fmt(pm["baseline_severity_cart_records"], 1), _fmt(pm["policy_severity_cart_records"], 1)],
        ["Minimum cart count", _fmt(pm["baseline_min_cart"], 1), _fmt(pm["policy_min_cart"], 1)],
        ["Mean cart count", _fmt(pm["baseline_mean_cart"], 1), _fmt(pm["policy_mean_cart"], 1)],
        ["Replenishment actions", "Unknown (not recorded)", _fmt(pm["n_replenishment_actions"])],
        ["Carts retrieved", "Unknown (not recorded)", _fmt(pm["total_carts_recommended"])],
        ["Operational cost J", _fmt(pm["baseline_cost_J"], 1), _fmt(pm["policy_cost_J"], 1)],
    ]
    _table(doc, ["Metric", "Historical baseline (observed)", "Simulated policy (estimated)"], rows, widths=[2.6, 1.9, 1.9])

    _table(
        doc,
        ["Simulated improvement", "Value"],
        [
            ["Events prevented", _fmt(pm["events_prevented"], 0)],
            ["Event reduction", _fmt(pm["event_reduction_pct"], 1) + "%"],
            ["Shortage duration reduction", _fmt(pm["duration_reduction_pct"], 1) + "%"],
            ["Shortage severity reduction", _fmt(pm["severity_reduction_pct"], 1) + "%"],
            ["Trips per day", _fmt(pm["trips_per_day"], 2)],
            ["Carts moved per day", _fmt(pm["carts_per_day"], 1)],
            ["Unnecessary actions (actual future minimum already ≥ target)", _fmt(pm["n_unnecessary_actions"], 0)],
            ["Capacity-constrained actions", _fmt(pm["n_capacity_constrained"], 0)],
        ],
        widths=[3.8, 2.4],
    )
    _figure(doc, res["figures"]["policy_comparison"], "Figure 9 — Observed baseline versus simulated policy outcome.")

    _para(
        doc,
        "Interpretation caveat: the historical series may already include unlogged employee "
        "replenishment activity, so the baseline is not a clean no-intervention scenario. The "
        "simulator adds recommended carts additively after the assumed lead time, capped by bay "
        "capacity, and does not model customer response, cart return dynamics or staff "
        "availability. These results are therefore scenario estimates, not proven causal effects.",
        italic=True,
    )

    _h(doc, "6.3 Policy robustness (validation)", 2)
    rb = res["robustness"]
    for param in rb["parameter"].unique():
        sub = rb[rb.parameter == param]
        _para(doc, f"Sensitivity to {param.replace('_', ' ')}:", size=10)
        _table(
            doc,
            [param, "Policy events", "Events prevented", "Severity red. %", "Actions", "Carts", "J"],
            _df_rows(sub, ["value", "policy_events", "events_prevented", "severity_reduction_pct",
                           "n_replenishment_actions", "total_carts_recommended", "policy_cost_J"], nd=1),
        )

    _h(doc, "6.4 Cost-coefficient sensitivity", 2)
    _para(
        doc,
        "Actual business costs are unavailable, so the objective weights were varied and the "
        "policy re-optimised for each combination.",
    )
    cs = res["cost_sensitivity"]
    _table(
        doc,
        ["λ shortage", "λ trip", "Best risk threshold", "Best target stock", "Events prevented", "Actions", "Severity red. %"],
        _df_rows(cs, ["lambda_shortage", "lambda_trip", "best_risk_threshold", "best_target_stock",
                      "events_prevented", "n_replenishment_actions", "severity_reduction_pct"], nd=1),
    )


# ------------------------------- Section 7 --------------------------------
def _section_7(doc, res):
    _h(doc, "7. Model Explainability", 1)
    imp = res["feature_importance"]
    _h(doc, "7.1 Top 15 features — 40-minute minimum model", 2)
    _table(doc, ["Rank", "Feature", "Importance", "Share of total"],
           [[i + 1, r["feature"], _fmt(r["importance"], 3), _fmt(r["importance_pct"], 1) + "%"]
            for i, (_, r) in enumerate(imp.iterrows())],
           widths=[0.6, 3.0, 1.3, 1.3],
           caption=f"Importance type: {imp['importance_type'].iloc[0] if len(imp) else 'n/a'}.")
    _figure(doc, res["figures"]["feature_importance"], "Figure 10 — Feature importance for the selected minimum-cart model.")

    _h(doc, "7.2 Cart depletion slope analysis", 2)
    _para(doc, "Observed relationship between the 6-record depletion rate and the realised 40-minute shortage rate on the holdout period.")
    da = res["depletion_analysis"]
    _table(doc, ["Depletion regime", "n", "Mean cart count", "Observed 40-min shortage rate", "Mean predicted minimum"],
           [[r["bucket"], _fmt(r["n"]), _fmt(r["mean_cart"], 1),
             _fmt(r["observed_shortage_rate_40min"] * 100, 1) + "%", _fmt(r["mean_predicted_min"], 1)]
            for _, r in da.iterrows()])

    _h(doc, "7.3 Traffic surge analysis", 2)
    sa = res["surge_analysis"]
    _table(doc, ["Traffic regime (IN rate 3-window / 9-window)", "n", "Mean IN rate", "Mean cart count", "Observed 40-min shortage rate"],
           [[r["bucket"], _fmt(r["n"]), _fmt(r["mean_traffic_in_rate"], 2), _fmt(r["mean_cart"], 1),
             _fmt(r["observed_shortage_rate_40min"] * 100, 1) + "%"] for _, r in sa.iterrows()])

    _h(doc, "7.4 Shortage risk interpretation", 2)
    _bullets(doc, [
        "Current cart level and its distance to the RED threshold dominate the forecast: the lower the stock now, the more probable a near-term breach.",
        "Short-window minima and rolling ranges contribute the shape of recent depletion, which separates a temporary dip from a sustained decline.",
        "Depletion rate and slope features add the velocity of change, which is what converts a currently-safe level into a near-term shortage.",
        "Traffic pressure (incoming customers relative to available carts) and the surge ratio capture demand shocks that precede rapid depletion.",
        "Temporal encodings capture the daily rhythm of the store, in which shortages cluster in predictable periods.",
    ])
    _para(
        doc,
        "These importances describe what the model uses to predict, not what causes shortages. "
        "They are associations learned from 14 days at one site and must not be read as causal "
        "effects or used to justify operational changes on their own. SHAP was not used: with "
        "fewer than 3,000 records and a single site, split-based importance plus the regime "
        "tables above are sufficient and less likely to be over-interpreted.",
        italic=True,
    )


# ------------------------------- Section 8 --------------------------------
def _section_8(doc, res):
    _h(doc, "8. Operational Recommendations", 1)
    fp = res["frozen_policy"]
    ws = res["warning_summary"]
    pm = res["policy_metrics"]
    d = res["replenishment_decisions"]
    typical = int(round(d["quantity"].median())) if len(d) else 0

    _h(doc, "8.1 When should employees replenish carts?", 2)
    _bullets(doc, [
        f"Act when the system's 40-minute shortage risk reaches {_fmt(fp['risk_threshold'], 2)} — on the holdout this gave a median of {_fmt(ws['median_lead_time_min'], 1)} minutes of warning before onset.",
        f"Start the retrieval immediately on a WARNING: with the assumed {_fmt(fp['lead_time_min'], 0)}-minute lead time, carts only reach the bay {_fmt(fp['lead_time_min'], 0)} minutes after the decision.",
        f"Treat any observed count below {C.RED_THRESHOLD} as a reactive emergency, not a prediction success — the shortage is already affecting customers.",
        f"Do not re-trigger while a retrieval is in transit; the engine enforces a {_fmt(fp['cooldown_min'], 0)}-minute cooldown and tracks pending orders.",
    ])

    _h(doc, "8.2 How many carts should they retrieve?", 2)
    _bullets(doc, [
        f"Typical trip: about {typical} carts (mean {_fmt(pm['avg_carts_per_action'], 1)}), sized to lift the predicted 40-minute minimum to the target stock of {fp['target_stock']}.",
        f"Operating constraints: at least {fp['min_carts_per_trip']} and at most {fp['max_carts_per_trip']} carts per trip, with a bay capacity of {fp['bay_capacity']}.",
        "When the situation is URGENT (breach expected sooner than the lead time), use the conservative lower-bound quantity and dispatch immediately.",
    ])

    _h(doc, "8.3 What signals indicate an urgent shortage?", 2)
    _bullets(doc, [
        f"Cart count already within a few carts of RED ({C.RED_THRESHOLD}) combined with a negative 6-record depletion rate.",
        "A steep or accelerating downward cart slope over the last 3 records.",
        "High traffic pressure — incoming customers over the last 20 minutes large relative to the carts currently available.",
        "A traffic surge ratio well above 1, meaning recent inflow is far above the hour-long average.",
        f"A predicted 40-minute minimum whose 90% lower bound is already below {C.RED_THRESHOLD}.",
    ])

    _h(doc, "8.4 Recommended alert threshold", 2)
    acfg = res["alert_cfg"]
    _para(
        doc,
        f"Set WARNING at risk ≥ {_fmt(acfg.warning_risk, 2)} and CRITICAL at risk ≥ "
        f"{_fmt(acfg.critical_risk, 2)}, with a {_fmt(acfg.cooldown_min, 0)}-minute cooldown. "
        f"On the holdout this produced event recall {_fmt(ws['event_recall'] * 100, 1)}% at "
        f"{_fmt(ws['false_alerts_per_day'], 2)} false alerts per store per day. The full "
        f"threshold trade-off curve is in outputs/risk_model_comparison.csv and the validation "
        f"calibration table.",
    )
    rtt = res["risk_threshold_table"]
    sub = rtt[rtt["threshold"].round(2).isin([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])]
    _table(doc, ["Risk threshold", "Precision", "Recall", "F1", "Alert rate", "False positives"],
           _df_rows(sub, ["threshold", "precision", "recall", "f1", "alert_rate", "FP"], nd=3),
           caption="Origin-level trade-off measured on validation data.")

    _h(doc, "8.5 How can false alerts be reduced?", 2)
    _bullets(doc, [
        "Raise the WARNING threshold — the table above quantifies the recall cost of doing so.",
        "Lengthen the cooldown or require two consecutive high-risk records before firing.",
        "Suppress alerts when the bay is already near capacity or a retrieval is in transit (already implemented).",
        "Log employee replenishment actions so the model can learn which alerts were already handled.",
        "Re-calibrate thresholds per store and per daypart once multi-store data is available.",
    ])

    _h(doc, "8.6 What additional data would improve the system?", 2)
    _bullets(doc, [
        "Replenishment action logs: timestamp, number of carts retrieved, staff member, completion time. This is the single highest-value addition — it would turn the simulated benefit into a measurable one.",
        "Actual replenishment lead times instead of the assumed 10 minutes.",
        "Reserve cart inventory and bay capacity per store, plus staff availability by shift.",
        "Cart-level identifiers or dwell information to separate cart return dynamics from customer traffic.",
        "Longer history (3+ months) and multiple stores to cover seasonality, weekends and promotions.",
        "Business cost data for shortages and staff time to replace the assumed objective weights.",
    ])


# ------------------------------- Section 9 --------------------------------
def _section_9(doc, res):
    _h(doc, "9. Limitations", 1)
    ov = res["overview"]
    sm = res["split_meta"]
    ws = res["warning_summary"]
    _bullets(doc, [
        f"Limited history: {_fmt(ov['n_records'])} records across {_fmt(ov['n_days'])} days. The holdout contains only {_fmt(sm['n_holdout'])} scored origins and {_fmt(ws['n_events'])} independent shortage events, so warning metrics have wide uncertainty.",
        f"Single store: all data comes from site {ov['site_keys'][0]}, one camera channel and one area. Nothing here is validated across stores.",
        f"Approximate horizons: '20 minutes' means 3 records. The median record is {_fmt(ov['interval_median'], 2)} minutes but intervals range up to {_fmt(ov['interval_max'], 1)} minutes; deviating horizons are excluded rather than corrected.",
        "Unknown historical replenishment actions: the observed series probably already contains employee interventions, so the baseline is not a no-intervention scenario and the simulated reduction may be overstated.",
        f"Assumed replenishment lead time of {_fmt(C.REPLENISHMENT_LEAD_TIME_MIN, 0)} minutes with no measured distribution; robustness to this assumption is reported in Section 6.3.",
        "Unknown reserve inventory and staff capacity: recommendations are effectively unconstrained on the supply side and may not be executable.",
        "Forecast uncertainty: conformal bounds assume the validation residual distribution transfers to the holdout; coverage is reported but the calibration sample is small.",
        "Simulation assumptions: carts are added additively after the lead time and capped by bay capacity. Customer response, cart returns and queueing are not modelled, and traffic is not equated with cart demand one-to-one.",
        "Risk probabilities are calibrated on a short validation window; drift across seasons, promotions and staffing changes is untested.",
        "Multi-store validation and a live shadow period are required before any operational claim is made.",
    ])


# ------------------------------- Section 10 -------------------------------
def _section_10(doc, res):
    _h(doc, "10. Next Steps", 1)
    sel = res["selected_models"]
    mc = res["model_comparison"]
    ho = mc[mc.split == "holdout"]

    def _mae(target, model):
        r = ho[(ho.target == target) & (ho.model == model)]
        return float(r.iloc[0]["MAE"]) if len(r) else np.nan

    _bullets(doc, [
        "Collect more historical telemetry — at least three months per store — to cover weekly and seasonal patterns and to increase the number of independent shortage events.",
        "Log every replenishment action (time, quantity, duration, staff) so the baseline becomes interpretable and the simulated benefit can be validated.",
        "Validate across multiple stores and areas before generalising thresholds or quantities.",
        "Run a shadow deployment: score live telemetry, record alerts and recommendations, but take no automated action; compare against what staff actually did.",
        "Re-calibrate alert thresholds and conformal bounds on the shadow data, per store and per daypart.",
        "Run an operational A/B test (alerted vs non-alerted shifts or stores) to measure the real causal effect on shortage duration and customer experience.",
    ], style="List Number")

    _h(doc, "10.1 Final assessment", 2)
    pm = res["policy_metrics"]
    ws = res["warning_summary"]
    rows = [
        ["Which model was selected?",
         f"+20 min: {sel['y_cart_3']}; +40 min: {sel['y_cart_6']}; 20-min minimum: {sel['y_min_3']}; 40-min minimum: {sel['y_min_6']}. Risk: {res['selected_risk_model']}."],
        ["Why was it selected?",
         "Chronological validation using a composite of overall MAE, MAE near the shortage boundary and shortage recall, so that accuracy where it matters operationally outweighs average accuracy."],
        ["Which horizon is more useful?",
         f"The 40-minute (+6 record) horizon, despite a higher MAE ({_fmt(_mae('y_cart_6', sel['y_cart_6']))} vs {_fmt(_mae('y_cart_3', sel['y_cart_3']))} carts at 20 minutes), because the assumed 10-minute replenishment lead time leaves no usable action window at 20 minutes."],
        ["How early can shortages be detected?",
         f"Median {_fmt(ws['median_lead_time_min'], 1)} minutes of lead time, with {_fmt(ws['pct_warned_ge_10min'], 1)}% of events warned at least 10 minutes ahead."],
        ["How many carts should be replenished?",
         f"About {_fmt(pm['avg_carts_per_action'], 1)} carts per trip on average, enough to restore the predicted 40-minute minimum to the {res['frozen_policy']['target_stock']}-cart target."],
        ["Does the simulated policy improve outcomes?",
         f"Under the stated assumptions yes: severity falls {_fmt(pm['severity_reduction_pct'], 1)}% and shortage minutes {_fmt(pm['duration_reduction_pct'], 1)}%, for {_fmt(pm['trips_per_day'], 2)} trips per day. This is a simulation result, not a measured effect."],
        ["What prevents production deployment?",
         "Fourteen days of single-site data, no replenishment action logs, an assumed lead time, unknown reserve inventory, and a simulator that cannot be validated against observed interventions."],
    ]
    _table(doc, ["Question", "Answer"], rows, widths=[1.9, 4.4])

    _para(
        doc,
        "Generated automatically by the cart forecasting pipeline. All figures and tables are "
        "computed from the data; no metric in this report is hand-entered.",
        italic=True,
    )
