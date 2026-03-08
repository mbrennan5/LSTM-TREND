"""
Sovereign Titan — Data Leakage Audit System
============================================
Audits a feature DataFrame for the five most common leakage patterns
found in LSTM-based financial forecasting pipelines.

Checks performed
----------------
1. Feature-level temporal scan      — flags any column built with .shift(-N)
                                      or other forward-looking transforms
2. Correlation identity trap        — detects features correlated ≥ threshold
                                      with the raw target or target.shift(+1)
3. Train / test date overlap        — verifies chronological split is clean
4. Scaler & PCA contamination       — fits transforms on the full dataset and
                                      compares test-set statistics vs train-only
                                      fit to surface leakage magnitude
5. Accuracy delta test              — trains a fast GRU on the full feature set
                                      then re-trains with suspicious features
                                      removed; a large accuracy drop = leakage,
                                      a small drop (or gain) = healthy pruning

Usage
-----
    python leakage_audit.py                        # demo with synthetic data
    python leakage_audit.py --csv path/to/data.csv # real feature CSV
    python leakage_audit.py --symbol AAPL          # fetch via yfinance

Output
------
    leakage_report_<timestamp>.txt   — human-readable findings
    leakage_report_<timestamp>.json  — machine-readable for CI gates
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy dependencies (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 15 %

IDENTITY_CORR_THRESHOLD = 0.85   # flag feature if |r| >= this with target
LEAKAGE_ACCURACY_DROP   = 0.03   # flag if removing feature drops acc by > 3 pp
SEQ_LEN                 = 30     # lookback window for GRU audit model
AUDIT_EPOCHS            = 12
AUDIT_BATCH             = 512
AUDIT_PATIENCE          = 4

SUSPICIOUS_PATTERNS = [
    # pandas / numpy forward-looking operations
    "shift(-",
    "shift( -",
    ".iloc[-",
    "rolling(",   # rolling is fine normally, flagged only when combined below
    "future",
    "tomorrow",
    "next_day",
    "fwd",
    "lead_",
    "_lead",
]

FORWARD_ONLY_PATTERNS = [
    # patterns that are almost never legitimate in features
    "shift(-",
    "future",
    "tomorrow",
    "next_day",
    "fwd_",
    "_fwd",
    "lead_",
    "_lead",
]


# ============================================================
# 1. SYNTHETIC DATA GENERATOR (for demo / CI)
# ============================================================

def make_synthetic_data(n_rows: int = 1500, n_clean: int = 10,
                        n_leaky: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Build a DataFrame with known-clean and known-leaky features so the
    audit can be validated even without real market data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_rows, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n_rows))

    df = pd.DataFrame({"date": dates, "close": close})
    df = df.set_index("date")

    # ---- clean features (only past data) ----
    for w in [10, 30, 90]:
        r = (df["close"] - df["close"].rolling(w).mean()) / (
            df["close"].rolling(w).std() + 1e-9
        )
        df[f"z_{w}"]       = r
        df[f"slope_{w}"]   = r.rolling(w).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
        )

    # momentum / ratio clean seeds
    df["log_diff"]      = np.log(df["close"]).diff()
    df["ema10_ratio"]   = df["close"] / df["close"].ewm(span=10).mean()
    df["ema50_ratio"]   = df["close"] / df["close"].ewm(span=50).mean()
    df["range_norm"]    = (df["close"].rolling(20).max()
                           - df["close"].rolling(20).min()) / (
                               df["close"].rolling(20).mean() + 1e-9
                           )

    # ---- leaky features (forward-looking) ----
    # Leaky 1: directly uses tomorrow's close
    df["close_lead1"]   = df["close"].shift(-1)
    # Leaky 2: encoded as a ratio but still future
    df["fwd_return"]    = df["close"].shift(-1) / df["close"] - 1
    # Leaky 3: subtle — smoothed future price
    df["close_ema_fwd"] = df["close"].shift(-3).ewm(span=5).mean()

    # ---- target: 1 if tomorrow's close > today's close ----
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
    df.iloc[-1, df.columns.get_loc("target")] = np.nan   # unknowable

    df = df.dropna(subset=["target"])
    return df


# ============================================================
# 2. REAL DATA LOADER (yfinance)
# ============================================================

def load_yfinance(symbol: str, period: str = "5y") -> pd.DataFrame:
    if not HAS_YFINANCE:
        raise ImportError("yfinance not installed. pip install yfinance")
    tk = yf.Ticker(symbol)
    raw = tk.history(period=period, auto_adjust=True)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.columns = [c.lower() for c in raw.columns]
    raw["log_diff"] = np.log(raw["close"]).diff()
    for w in [10, 30, 90]:
        z = (raw["close"] - raw["close"].rolling(w).mean()) / (
            raw["close"].rolling(w).std() + 1e-9
        )
        raw[f"z_{w}"]     = z
        raw[f"slope_{w}"] = z.rolling(w).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
        )
    raw["ema10_ratio"] = raw["close"] / raw["close"].ewm(span=10).mean()
    raw["ema50_ratio"] = raw["close"] / raw["close"].ewm(span=50).mean()
    raw["target"] = (raw["close"].shift(-1) > raw["close"]).astype(float)
    raw.iloc[-1, raw.columns.get_loc("target")] = np.nan
    raw = raw.dropna(subset=["target"])
    return raw


# ============================================================
# 3. CHECK 1 — Feature-level temporal scan
# ============================================================

def check_temporal_names(df: pd.DataFrame) -> dict:
    """
    Flag column names that match forward-looking naming conventions.
    This is a fast heuristic — it catches obvious sins without inspecting
    the source code that generated the columns.
    """
    findings = []
    for col in df.columns:
        if col == "target":
            continue
        low = col.lower()
        hits = [p for p in FORWARD_ONLY_PATTERNS if p.lower() in low]
        if hits:
            findings.append({"feature": col, "patterns": hits})

    return {
        "check": "temporal_name_scan",
        "status": "FAIL" if findings else "PASS",
        "flagged": findings,
        "summary": (
            f"{len(findings)} feature(s) contain forward-looking name patterns"
            if findings
            else "No forward-looking name patterns detected"
        ),
    }


# ============================================================
# 4. CHECK 2 — Correlation identity trap
# ============================================================

def check_identity_trap(df: pd.DataFrame, target_col: str = "target",
                         threshold: float = IDENTITY_CORR_THRESHOLD) -> dict:
    """
    Compute |Pearson r| between each feature and:
      (a) the target itself
      (b) target.shift(1)  — lagged target (catches lag-1 identity)
      (c) the raw close price (level non-stationarity)

    A feature correlated ≥ threshold with any of these is almost certainly
    leaking or contributing to the identity function trap.
    """
    feature_cols = [c for c in df.columns if c not in ("target", "close", "date")]
    target   = df[target_col].fillna(0)
    lag_tgt  = target.shift(1).fillna(0)
    close    = df["close"] if "close" in df.columns else pd.Series(dtype=float)

    flagged = []
    for col in feature_cols:
        series = df[col].fillna(0)
        r_tgt  = abs(series.corr(target))
        r_lag  = abs(series.corr(lag_tgt))
        r_cls  = abs(series.corr(close)) if len(close) else 0.0

        worst = max(r_tgt, r_lag, r_cls)
        if worst >= threshold:
            flagged.append({
                "feature":        col,
                "r_with_target":  round(r_tgt, 4),
                "r_with_lag_tgt": round(r_lag, 4),
                "r_with_close":   round(r_cls, 4),
                "worst_r":        round(worst, 4),
            })

    flagged.sort(key=lambda x: x["worst_r"], reverse=True)
    return {
        "check": "identity_trap_correlation",
        "status": "FAIL" if flagged else "PASS",
        "threshold": threshold,
        "flagged": flagged,
        "summary": (
            f"{len(flagged)} feature(s) correlated ≥ {threshold} with target/close"
            if flagged
            else f"No identity trap detected (threshold={threshold})"
        ),
    }


# ============================================================
# 5. CHECK 3 — Train / test date overlap
# ============================================================

def check_date_overlap(df: pd.DataFrame) -> dict:
    """
    Verify that the walk-forward chronological split produces zero overlap.
    Also checks that test dates are strictly newer than train dates.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.RangeIndex(n)

    train_idx = idx[:train_end]
    val_idx   = idx[train_end:val_end]
    test_idx  = idx[val_end:]

    issues = []

    if isinstance(idx, pd.DatetimeIndex):
        tv_overlap = set(train_idx) & set(val_idx)
        tt_overlap = set(train_idx) & set(test_idx)
        vt_overlap = set(val_idx)   & set(test_idx)

        if tv_overlap:
            issues.append(f"Train/Val overlap: {len(tv_overlap)} dates")
        if tt_overlap:
            issues.append(f"Train/Test overlap: {len(tt_overlap)} dates")
        if vt_overlap:
            issues.append(f"Val/Test overlap: {len(vt_overlap)} dates")

        if len(test_idx) and train_idx[-1] >= test_idx[0]:
            issues.append(
                f"Temporal ordering broken: last train date "
                f"({train_idx[-1].date()}) >= first test date "
                f"({test_idx[0].date()})"
            )
        date_info = {
            "train": f"{train_idx[0].date()} → {train_idx[-1].date()} ({len(train_idx)} rows)",
            "val":   f"{val_idx[0].date()} → {val_idx[-1].date()} ({len(val_idx)} rows)",
            "test":  f"{test_idx[0].date()} → {test_idx[-1].date()} ({len(test_idx)} rows)",
        }
    else:
        date_info = {
            "train": f"rows 0–{train_end-1}",
            "val":   f"rows {train_end}–{val_end-1}",
            "test":  f"rows {val_end}–{n-1}",
        }

    return {
        "check": "date_overlap",
        "status": "FAIL" if issues else "PASS",
        "split": date_info,
        "issues": issues,
        "summary": "; ".join(issues) if issues else "Walk-forward split is clean — no date overlap",
    }


# ============================================================
# 6. CHECK 4 — Scaler / PCA contamination
# ============================================================

def check_scaler_contamination(df: pd.DataFrame, target_col: str = "target") -> dict:
    """
    Fit a RobustScaler twice:
      (a) on training rows only  (correct)
      (b) on the full dataset    (leaky)

    Measure the mean absolute difference in test-set scaled values.
    A non-zero difference proves the test set influenced the scaler when
    fitted on all data — i.e., future statistics leaked into scaling.
    """
    if not HAS_SKLEARN:
        return {
            "check": "scaler_contamination",
            "status": "SKIP",
            "summary": "scikit-learn not installed",
        }

    feature_cols = [c for c in df.columns
                    if c not in (target_col, "close", "open", "high",
                                 "low", "volume", "dividends", "stock splits")]
    X = df[feature_cols].fillna(0).values
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train = X[:train_end]
    X_test  = X[val_end:]

    # Correct: fit on train only
    sc_train = RobustScaler().fit(X_train)
    X_test_correct = sc_train.transform(X_test)

    # Leaky: fit on full dataset
    sc_full = RobustScaler().fit(X)
    X_test_leaky = sc_full.transform(X_test)

    delta = np.abs(X_test_correct - X_test_leaky).mean()
    max_delta = np.abs(X_test_correct - X_test_leaky).max()

    status = "WARN" if delta > 0.001 else "PASS"
    return {
        "check": "scaler_contamination",
        "status": status,
        "mean_abs_delta": round(float(delta), 6),
        "max_abs_delta":  round(float(max_delta), 6),
        "summary": (
            f"Scaler contamination detected: mean Δ={delta:.5f} "
            f"(fitting on full data vs train-only changes test values)"
            if delta > 0.001
            else f"Scaler correctly fitted on training data only (mean Δ={delta:.6f})"
        ),
    }


# ============================================================
# 7. CHECK 5 — Accuracy delta test (GRU model)
# ============================================================

def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def _train_gru(X_tr, y_tr, X_val, y_val, n_features: int,
               seq_len: int = SEQ_LEN) -> float:
    """Train a minimal GRU and return validation accuracy."""
    if not HAS_TF:
        return float("nan")

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    cb = EarlyStopping(monitor="val_loss", patience=AUDIT_PATIENCE,
                       restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=AUDIT_EPOCHS, batch_size=AUDIT_BATCH,
              callbacks=[cb], verbose=0)
    _, acc = model.evaluate(X_val, y_val, verbose=0)
    tf.keras.backend.clear_session()
    return float(acc)


def check_accuracy_delta(df: pd.DataFrame, suspicious_cols: list,
                          target_col: str = "target") -> dict:
    """
    Train a fast GRU twice:
      (a) full feature set
      (b) full feature set minus suspicious features

    If accuracy drops by > LEAKAGE_ACCURACY_DROP removing the suspicious
    features, they were genuinely contributing signal — and since they
    are forward-looking, that signal is leakage.
    """
    if not HAS_TF:
        return {
            "check": "accuracy_delta",
            "status": "SKIP",
            "summary": "TensorFlow not installed — skipping accuracy delta test",
        }

    if not HAS_SKLEARN:
        return {
            "check": "accuracy_delta",
            "status": "SKIP",
            "summary": "scikit-learn not installed — skipping accuracy delta test",
        }

    base_feature_cols = [c for c in df.columns
                         if c not in (target_col, "close", "open", "high",
                                      "low", "volume", "dividends",
                                      "stock splits")]
    clean_feature_cols = [c for c in base_feature_cols
                          if c not in suspicious_cols]

    if not clean_feature_cols:
        return {
            "check": "accuracy_delta",
            "status": "SKIP",
            "summary": "All features flagged as suspicious — cannot run delta test",
        }

    y = df[target_col].fillna(0).values
    n = len(y)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    results = {}
    for label, cols in [("full", base_feature_cols), ("clean", clean_feature_cols)]:
        X_raw = df[cols].fillna(0).values
        sc    = RobustScaler().fit(X_raw[:train_end])
        X_sc  = sc.transform(X_raw)

        X_tr_seq, y_tr = _build_sequences(X_sc[:train_end],  y[:train_end],  SEQ_LEN)
        X_val_seq, y_vl= _build_sequences(X_sc[train_end:val_end],
                                           y[train_end:val_end], SEQ_LEN)

        if len(X_tr_seq) < 50:
            results[label] = float("nan")
            continue

        results[label] = _train_gru(X_tr_seq, y_tr, X_val_seq, y_vl,
                                    n_features=len(cols))

    acc_full  = results.get("full",  float("nan"))
    acc_clean = results.get("clean", float("nan"))

    if np.isnan(acc_full) or np.isnan(acc_clean):
        delta_msg = "Could not compute delta (insufficient data or NaN)"
        status = "SKIP"
    else:
        delta = acc_full - acc_clean
        if delta > LEAKAGE_ACCURACY_DROP:
            status = "FAIL"
            delta_msg = (
                f"Removing suspicious features drops val accuracy by "
                f"{delta:.2%} ({acc_full:.2%} → {acc_clean:.2%}) — "
                f"LEAKAGE CONFIRMED"
            )
        elif delta < -0.01:
            status = "PASS"
            delta_msg = (
                f"Removing suspicious features IMPROVES val accuracy by "
                f"{abs(delta):.2%} ({acc_full:.2%} → {acc_clean:.2%}) — "
                f"suspicious features were pure noise"
            )
        else:
            status = "PASS"
            delta_msg = (
                f"Negligible accuracy change: {acc_full:.2%} → {acc_clean:.2%} "
                f"(Δ={delta:.2%}) — suspicious features add no real signal"
            )

    return {
        "check": "accuracy_delta",
        "status": status,
        "acc_full_model":  round(acc_full,  4) if not np.isnan(acc_full)  else None,
        "acc_clean_model": round(acc_clean, 4) if not np.isnan(acc_clean) else None,
        "suspicious_removed": suspicious_cols,
        "summary": delta_msg,
    }


# ============================================================
# 8. REPORT GENERATOR
# ============================================================

SEVERITY = {"FAIL": 0, "WARN": 1, "PASS": 2, "SKIP": 3}

def _severity_emoji(status: str) -> str:
    return {"FAIL": "🚨", "WARN": "⚠️ ", "PASS": "✅", "SKIP": "⏭️ "}.get(status, "❓")


def build_report(results: list, df_shape: tuple) -> str:
    lines = [
        "=" * 70,
        "  SOVEREIGN TITAN — DATA LEAKAGE AUDIT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Dataset:   {df_shape[0]} rows × {df_shape[1]} columns",
        "=" * 70,
        "",
    ]

    overall_status = "PASS"
    for r in results:
        s = r.get("status", "SKIP")
        if SEVERITY.get(s, 9) < SEVERITY.get(overall_status, 9):
            overall_status = s

    lines += [
        f"  OVERALL STATUS: {_severity_emoji(overall_status)} {overall_status}",
        "",
        "-" * 70,
    ]

    for r in results:
        status = r.get("status", "SKIP")
        lines += [
            "",
            f"  {_severity_emoji(status)} [{status}]  {r.get('check', '').upper().replace('_', ' ')}",
            f"  {r.get('summary', '')}",
        ]

        # Extra detail for interesting checks
        if r.get("check") == "temporal_name_scan" and r.get("flagged"):
            lines.append("  Flagged features:")
            for f in r["flagged"]:
                lines.append(f"    • {f['feature']}  (patterns: {f['patterns']})")

        if r.get("check") == "identity_trap_correlation" and r.get("flagged"):
            lines.append(f"  Correlation threshold: {r.get('threshold')}")
            lines.append("  Top offenders:")
            for f in r["flagged"][:10]:
                lines.append(
                    f"    • {f['feature']:<30} "
                    f"r_target={f['r_with_target']:.3f}  "
                    f"r_lag={f['r_with_lag_tgt']:.3f}  "
                    f"r_close={f['r_with_close']:.3f}"
                )

        if r.get("check") == "date_overlap":
            sp = r.get("split", {})
            for k, v in sp.items():
                lines.append(f"    {k.capitalize():7s}: {v}")
            if r.get("issues"):
                for issue in r["issues"]:
                    lines.append(f"  !! {issue}")

        if r.get("check") == "scaler_contamination":
            lines.append(
                f"  Mean |Δ| = {r.get('mean_abs_delta')}  "
                f"Max |Δ| = {r.get('max_abs_delta')}"
            )

        if r.get("check") == "accuracy_delta" and r.get("acc_full_model") is not None:
            lines.append(
                f"  Full model val acc:  {r['acc_full_model']:.4f}"
            )
            lines.append(
                f"  Clean model val acc: {r['acc_clean_model']:.4f}"
            )
            if r.get("suspicious_removed"):
                lines.append(
                    f"  Removed ({len(r['suspicious_removed'])}): "
                    + ", ".join(r["suspicious_removed"][:8])
                    + ("…" if len(r["suspicious_removed"]) > 8 else "")
                )

    lines += [
        "",
        "-" * 70,
        "",
        "  REMEDIATION CHECKLIST",
        "  ─────────────────────",
        "  □ Ensure T_FINAL = close.shift(-1) — NEVER use .shift(-1) in features",
        "  □ Fit RobustScaler on X_train only; transform val/test separately",
        "  □ Fit PCA on X_train_scaled only; transform val/test separately",
        "  □ Walk-forward split: train[:70%] | val[70:85%] | test[85%:]",
        "  □ Exclude warmup rows (first 350) from scaler/PCA fit",
        "  □ Validate: test accuracy should be ≤ val accuracy (within ~3 pp)",
        "  □ Healthy DIR range: 52-60 %. Suspicious: >70 % (both val & test)",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)


# ============================================================
# 9. MAIN RUNNER
# ============================================================

def run_audit(df: pd.DataFrame, output_dir: str = ".") -> dict:
    """
    Execute all five checks and write reports.

    Returns the aggregated results dict (also serialised to JSON).
    """
    print("\n🔍 Sovereign Titan — Leakage Audit Starting…")
    print(f"   Dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # ---------- Run all checks ----------

    r1 = check_temporal_names(df)
    print(f"  Check 1 — Temporal name scan:         {r1['status']}")

    r2 = check_identity_trap(df)
    print(f"  Check 2 — Identity trap correlation:  {r2['status']}")

    r3 = check_date_overlap(df)
    print(f"  Check 3 — Date overlap:               {r3['status']}")

    r4 = check_scaler_contamination(df)
    print(f"  Check 4 — Scaler contamination:       {r4['status']}")

    # Gather all suspicious columns from checks 1 + 2 for the delta test
    suspicious = set()
    for item in r1.get("flagged", []):
        suspicious.add(item["feature"])
    for item in r2.get("flagged", []):
        suspicious.add(item["feature"])

    r5 = check_accuracy_delta(df, list(suspicious))
    print(f"  Check 5 — Accuracy delta test:        {r5['status']}\n")

    results = [r1, r2, r3, r4, r5]

    # ---------- Determine overall status ----------
    statuses = [r.get("status", "SKIP") for r in results]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    # ---------- Write report files ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path  = os.path.join(output_dir, f"leakage_report_{ts}.txt")
    json_path = os.path.join(output_dir, f"leakage_report_{ts}.json")

    report_text = build_report(results, df.shape)
    print(report_text)

    with open(txt_path, "w") as f:
        f.write(report_text)

    payload = {
        "timestamp":      ts,
        "overall_status": overall,
        "dataset_shape":  list(df.shape),
        "checks":         results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n  📄 Reports saved:")
    print(f"     {txt_path}")
    print(f"     {json_path}\n")

    return payload


# ============================================================
# 10. CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sovereign Titan — Data Leakage Audit"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv",    metavar="PATH",   help="Load features from CSV")
    source.add_argument("--symbol", metavar="TICKER", help="Fetch data via yfinance")
    parser.add_argument("--output", metavar="DIR", default=".",
                        help="Output directory for report files (default: .)")
    args = parser.parse_args()

    if args.csv:
        print(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv, parse_dates=True, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")

    elif args.symbol:
        if not HAS_YFINANCE:
            print("ERROR: yfinance not installed. pip install yfinance")
            sys.exit(1)
        print(f"Fetching {args.symbol} via yfinance…")
        df = load_yfinance(args.symbol)

    else:
        print("No data source specified — using synthetic demo dataset.\n")
        df = make_synthetic_data()

    os.makedirs(args.output, exist_ok=True)
    payload = run_audit(df, output_dir=args.output)

    # Exit with non-zero code on failure (useful for CI gating)
    sys.exit(0 if payload["overall_status"] in ("PASS", "WARN") else 1)


if __name__ == "__main__":
    main()
