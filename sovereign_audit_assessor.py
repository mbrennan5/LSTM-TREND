# ==============================================================================
# SOVEREIGN AUDIT ASSESSOR
# Standalone analysis tool for Sovereign_Audit_Master.csv
# Usage:
#   python sovereign_audit_assessor.py <path_to_csv>
#   python sovereign_audit_assessor.py                   # auto-finds latest CSV
# ==============================================================================

import os
import sys
import re
import glob
import argparse
import textwrap
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── colour helpers (ANSI, disabled on Windows if needed) ──────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"

def h(text): return f"{BOLD}{CYAN}{text}{RESET}"
def ok(text): return f"{GREEN}{text}{RESET}"
def warn(text): return f"{YELLOW}{text}{RESET}"
def bad(text): return f"{RED}{text}{RESET}"

DIVIDER  = "═" * 70
DIVIDER2 = "─" * 70

# ==============================================================================
# INDICATOR GROUP TAXONOMY  (matches sovereign_titan_v3_19_18.py feature set)
# ==============================================================================
#   LENS_XX features  — z-scored then slope/sos transformed
#   WIN_XX  features  — rolling % deviation (suffix = pct)
# Groups reflect the input signal type, not the transform applied.
INDICATOR_GROUPS = {
    "Bounded Oscillators": {
        "er", "vidya_cmo", "r_sq", "hurst", "shannon", "adx",
        "logistic_prob", "aroon_up", "donchian_high", "dispersion", "lr_slope",
    },
    "Price MA Ratios": {"tema", "sma", "hma", "kalman"},
    "Price-Unit Osc":  {"mtsi"},
    "WIN Rolling COG": {"cog"},
    # ── Log-Ratio Protocol (v4.0 mandate) ──────────────────────────────────────
    "LR Price Anchors":  {"kalman_sma_ratio", "tema_kalman_ratio", "exhaustion"},
    "LR Velocity Seeds": {"ratio_acc", "ratio_snr", "curvature_diff", "cycle_vs_trend"},
    "LR Interactions":   {
        "ratio_eff_slope", "ratio_pers_slope", "ratio_struct",
        "adx_entropy_ratio", "ratio_breakout_eff",
    },
}

# Physics Suite classification by transform suffix
# Position  → _z       (where is price/signal relative to its mean?)
# Velocity  → _z_slope (how fast is it moving?)
# Accel     → _z_sos   (is the velocity itself accelerating?)
PHYSICS_SUITE = {
    "z":       "Position / Z",
    "z_slope": "Velocity / Slope",
    "z_sos":   "Acceleration / SOS",
    "pct":     "WIN Rolling %",
}
# reverse lookup: family → group label
_FAM_TO_GROUP = {fam: grp for grp, fams in INDICATOR_GROUPS.items() for fam in fams}


# ==============================================================================
# FEATURE NAME PARSER
# ==============================================================================
_LENS_RE    = re.compile(r"^LENS_(\d+)_", re.IGNORECASE)
_WIN_RE     = re.compile(r"^WIN_(\d+)_",  re.IGNORECASE)
_SUFFIX_RE  = re.compile(r"_(z(?:_slope|_sos)?|pct)$", re.IGNORECASE)

def parse_feature(name: str) -> dict:
    """Break a feature column name into its structural parts."""
    d = {"name": name, "lens": None, "window": None,
         "family": None, "period": None, "suffix": None}

    m_lens = _LENS_RE.match(name)
    m_win  = _WIN_RE.match(name)

    if m_lens:
        d["lens"]   = int(m_lens.group(1))
        body        = name[m_lens.end():]          # strip LENS_XX_
    elif m_win:
        d["window"] = int(m_win.group(1))
        body        = name[m_win.end():]
    else:
        body = name

    # strip trailing _z / _z_slope / _z_sos
    m_suf = _SUFFIX_RE.search(body)
    if m_suf:
        d["suffix"] = m_suf.group(1).lower()
        body        = body[:m_suf.start()]

    # trailing numeric period
    m_per = re.search(r"_(\d+)$", body)
    if m_per:
        d["period"] = int(m_per.group(1))
        body        = body[:m_per.start()]

    d["family"] = body.lower()
    return d


# ==============================================================================
# LOAD & VALIDATE
# ==============================================================================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Brain", "Feature", "I_raw", "I_Norm"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(bad(f"CSV missing required columns: {missing}"))

    # optional but expected columns — fill with sensible defaults
    for col, default in [("Persistence", 1), ("A_Impact", df["I_Norm"]),
                         ("A_UV", np.nan), ("UV%", np.nan),
                         ("Max_R", np.nan), ("Is_Locked", False),
                         ("Iteration", 1), ("Model_Type", "LSTM")]:
        if col not in df.columns:
            df[col] = default

    # attach parsed fields
    parsed        = df["Feature"].apply(parse_feature).apply(pd.Series)
    df            = pd.concat([df, parsed[["lens","window","family","period","suffix"]]], axis=1)
    df["src_type"] = df.apply(
        lambda r: f"LENS_{int(r['lens'])}" if pd.notna(r["lens"])
                  else (f"WIN_{int(r['window'])}" if pd.notna(r["window"]) else "WIN"),
        axis=1
    )

    return df


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def _compute_sovereign_score(grp: pd.DataFrame, num_iters: int) -> pd.DataFrame:
    """
    70/15/15 Sovereign Score:
      70% — A_Impact   (normalised avg I_Norm — predictive contribution)
      15% — Persistence (fraction of iterations survived — regime-invariance)
      15% — A_UV       (avg uniqueness % / 100 — orthogonality insurance)

    Each component is min-max normalised to [0, 1] before weighting so that
    a high-persistence feature with zero impact cannot crowd out a lower-persistence
    feature with strong signal (prevents 'Information Bullies').
    """
    g = grp.copy()

    # ── Component 1: A_Impact (normalised avg I_Norm) ─────────────────────────
    ai = g["Avg_INorm"].fillna(0.0)
    ai_range = ai.max() - ai.min()
    g["C_Impact"] = (ai - ai.min()) / (ai_range + 1e-9)

    # ── Component 2: Persistence fraction ────────────────────────────────────
    g["C_Persist"] = (g["Persistence"].fillna(1) / max(num_iters, 1)).clip(0, 1)

    # ── Component 3: Avg UV% normalised ──────────────────────────────────────
    uv = g["Avg_UV"].fillna(50.0) / 100.0
    g["C_UV"] = uv.clip(0, 1)

    g["Sov_Score"] = 0.70 * g["C_Impact"] + 0.15 * g["C_Persist"] + 0.15 * g["C_UV"]
    return g


# ==============================================================================
# PER-BRAIN REPORT
# ==============================================================================
def _top_features(brain_df: pd.DataFrame, n: int = 19,
                  num_iters: int = 1) -> pd.DataFrame:
    """Aggregate by Feature, apply 70/15/15 scoring, return top-n."""
    grp = (brain_df.groupby("Feature")
           .agg(
               Runs       = ("I_raw",       "count"),
               Avg_IRaw   = ("I_raw",       "mean"),
               Avg_INorm  = ("I_Norm",      "mean"),
               Avg_UV     = ("UV%",         "mean"),
               Max_R_avg  = ("Max_R",       "mean"),
               Persistence= ("Persistence", "first"),
               Is_Locked  = ("Is_Locked",   "first"),
               Model_Type = ("Model_Type",  "first"),
           )
           .reset_index())
    grp = _compute_sovereign_score(grp, num_iters)
    return (grp.sort_values("Sov_Score", ascending=False)
               .head(n)
               .reset_index(drop=True))


def report_brain(brain: str, bdf: pd.DataFrame, num_iters: int):
    top = _top_features(bdf, num_iters=num_iters)
    n_feat   = bdf["Feature"].nunique()
    n_rows   = len(bdf)
    avg_imp  = bdf["I_Norm"].mean()
    avg_uv   = bdf["UV%"].mean() if bdf["UV%"].notna().any() else float("nan")

    print(f"\n{DIVIDER}")
    print(h(f"  BRAIN: {brain}   ({n_feat} unique features across {n_rows} rows)"))
    print(DIVIDER)
    print(f"  Avg I_Norm  : {avg_imp:.4f}    Avg UV%: {avg_uv:.1f}%"
          if not np.isnan(avg_uv) else f"  Avg I_Norm  : {avg_imp:.4f}")
    print()

    # header
    hdr = (f"  {'RNK':<4} {'FEATURE':<40} {'PERSIST':>7} "
           f"{'AVG_IMP':>8} {'AVG_UV%':>8} {'SOV_715':>8} {'LK':>4}")
    print(hdr)
    print("  " + "─" * 78)

    for i, row in top.iterrows():
        lock_icon = "🔒" if row["Is_Locked"] else "  "
        persist_s = f"{int(row['Persistence'])}/{num_iters}"
        color     = GREEN if row["Is_Locked"] else RESET
        uv_s      = f"{row['Avg_UV']:.1f}" if pd.notna(row["Avg_UV"]) else "  —  "
        line = (f"  {i+1:02d}.  {color}{row['Feature']:<40}{RESET} "
                f"{persist_s:>7}  {row['Avg_INorm']:>8.4f}  "
                f"{uv_s:>8}  {row['Sov_Score']:>8.4f}  {lock_icon}")
        print(line)

    print()

    # ── family breakdown ──────────────────────────────────────────────────────
    fam_counts = bdf.groupby("family")["I_Norm"].mean().sort_values(ascending=False)
    print(h("  Family breakdown (avg I_Norm):"))
    for fam, val in fam_counts.items():          # all families, not just top 10
        grp  = _FAM_TO_GROUP.get(fam, "Other")
        bar  = "█" * int(val * 40)
        print(f"    {fam:<28} {val:.4f}  [{grp}]  {bar}")

    # ── period breakdown ──────────────────────────────────────────────────────
    per_df = bdf[bdf["period"].notna()]
    if not per_df.empty:
        print()
        print(h("  Period breakdown (avg I_Norm per indicator period):"))
        per_grp = (per_df.groupby(["family","period"])["I_Norm"]
                         .mean()
                         .reset_index()
                         .sort_values("I_Norm", ascending=False))
        for _, row in per_grp.iterrows():
            print(f"    {row['family']:<22}  period={int(row['period']):>3}  "
                  f"avg={row['I_Norm']:.4f}")

    # ── LENS split ────────────────────────────────────────────────────────────
    if bdf["src_type"].notna().any():
        print()
        print(h("  LENS split:"))
        lens_grp = (bdf.groupby("src_type")
                    .agg(Features=("Feature","nunique"),
                         Avg_Imp=("I_Norm","mean"))
                    .reset_index()
                    .sort_values("Avg_Imp", ascending=False))
        for _, r in lens_grp.iterrows():
            print(f"    {r['src_type']:<12}  {r['Features']:>3} features  "
                  f"avg_imp={r['Avg_Imp']:.4f}")

    # ── suffix split ──────────────────────────────────────────────────────────
    if bdf["suffix"].notna().any():
        print()
        print(h("  Suffix split (transformation type):"))
        suf_grp = (bdf.groupby("suffix")
                   .agg(Cnt=("Feature","count"),
                        Avg_Imp=("I_Norm","mean"))
                   .reset_index()
                   .sort_values("Avg_Imp", ascending=False))
        for _, r in suf_grp.iterrows():
            print(f"    {str(r['suffix']):<14}  cnt={r['Cnt']:>4}  "
                  f"avg_imp={r['Avg_Imp']:.4f}")

    return top


# ==============================================================================
# INPUT GROUP ANALYSIS
# ==============================================================================
def report_input_groups(df: pd.DataFrame):
    """Break down importance by indicator group, LENS/WIN bucket, and period."""
    print(f"\n{DIVIDER}")
    print(h("  INPUT GROUP ANALYSIS"))
    print(DIVIDER)

    df2 = df.copy()
    df2["group"] = df2["family"].map(_FAM_TO_GROUP).fillna("Other")

    # ── Group-level summary ───────────────────────────────────────────────────
    grp_stats = (df2.groupby("group")["I_Norm"]
                    .agg(Count="count", Avg="mean", Max="max")
                    .sort_values("Avg", ascending=False))
    print(h("\n  Indicator group  (avg I_Norm):"))
    print(f"  {'GROUP':<24} {'COUNT':>6} {'AVG_IMP':>9} {'MAX_IMP':>9}")
    print("  " + "─" * 52)
    for grp, row in grp_stats.iterrows():
        bar = "█" * int(row["Avg"] * 40)
        print(f"  {grp:<24} {int(row['Count']):>6}  {row['Avg']:>8.4f}  "
              f"{row['Max']:>8.4f}  {bar}")

    # ── LENS_10 vs LENS_90 for bounded / MA families ──────────────────────────
    lens_df = df2[df2["lens"].notna()]
    if not lens_df.empty:
        print(h("\n  LENS_10 vs LENS_90 (bounded oscillators + price MA ratios):"))
        lens_cmp = (lens_df.groupby(["group", "src_type"])["I_Norm"]
                           .mean()
                           .unstack("src_type", fill_value=float("nan"))
                           .sort_values("LENS_10" if "LENS_10" in
                                        lens_df["src_type"].unique() else
                                        lens_df["src_type"].iloc[0],
                                        ascending=False, na_position="last"))
        cols = sorted(lens_cmp.columns)
        header = f"  {'GROUP':<24}" + "".join(f"  {c:>10}" for c in cols)
        print(header)
        print("  " + "─" * (24 + 12 * len(cols)))
        for grp, row in lens_cmp.iterrows():
            vals = "".join(
                f"  {row[c]:>10.4f}" if not pd.isna(row.get(c, float("nan"))) else "         —"
                for c in cols)
            print(f"  {grp:<24}{vals}")

    # ── WIN window split ──────────────────────────────────────────────────────
    win_df = df2[df2["window"].notna()]
    if not win_df.empty:
        print(h("\n  WIN rolling (COG) — by window size:"))
        win_cmp = (win_df.groupby("src_type")["I_Norm"]
                         .agg(Count="count", Avg="mean")
                         .sort_values("Avg", ascending=False))
        for src, row in win_cmp.iterrows():
            bar = "█" * int(row["Avg"] * 40)
            print(f"    {src:<12}  cnt={int(row['Count']):>4}  avg={row['Avg']:.4f}  {bar}")

    # ── Period breakdown ──────────────────────────────────────────────────────
    per_df = df2[df2["period"].notna()]
    if not per_df.empty:
        print(h("\n  Period breakdown (avg I_Norm per indicator period):"))
        per_cmp = (per_df.groupby(["family", "period"])["I_Norm"]
                         .mean()
                         .reset_index()
                         .sort_values("I_Norm", ascending=False))
        for _, row in per_cmp.iterrows():
            print(f"    {row['family']:<22}  period={int(row['period']):>3}  "
                  f"avg={row['I_Norm']:.4f}")

    # ── Suffix / transform breakdown ──────────────────────────────────────────
    suf_df = df2[df2["suffix"].notna()]
    if not suf_df.empty:
        print(h("\n  Transform suffix (avg I_Norm):"))
        suf_cmp = (suf_df.groupby("suffix")["I_Norm"]
                         .agg(Count="count", Avg="mean")
                         .sort_values("Avg", ascending=False))
        for suf, row in suf_cmp.iterrows():
            bar = "█" * int(row["Avg"] * 40)
            print(f"    _{suf:<14}  cnt={int(row['Count']):>4}  avg={row['Avg']:.4f}  {bar}")


# ==============================================================================
# CROSS-BRAIN ANALYSIS
# ==============================================================================
def report_cross_brain(df: pd.DataFrame, brains: list):
    print(f"\n{DIVIDER}")
    print(h("  CROSS-BRAIN ANALYSIS"))
    print(DIVIDER)

    # features appearing in multiple brains
    feat_brains = df.groupby("Feature")["Brain"].unique().reset_index()
    feat_brains["n_brains"] = feat_brains["Brain"].apply(len)
    shared = feat_brains[feat_brains["n_brains"] > 1].sort_values("n_brains", ascending=False)

    if shared.empty:
        print(warn("  No features appear in multiple brains."))
    else:
        print(f"\n  {len(shared)} feature(s) shared across brains:")
        print(f"  {'FEATURE':<42} {'BRAINS'}")
        print("  " + "─" * 60)
        for _, row in shared.iterrows():
            print(f"  {row['Feature']:<42} {', '.join(row['Brain'])}")

    # brain-to-brain impact divergence
    if len(brains) >= 2:
        print()
        print(h("  Impact divergence between brains:"))
        pivot = (df.groupby(["Brain", "Feature"])["I_Norm"].mean()
                   .unstack("Brain", fill_value=0))
        for i, b1 in enumerate(brains):
            for b2 in brains[i+1:]:
                if b1 in pivot.columns and b2 in pivot.columns:
                    diff = (pivot[b1] - pivot[b2]).abs().mean()
                    print(f"    {b1} ↔ {b2}  mean |ΔI_Norm| = {diff:.4f}")


# ==============================================================================
# ITERATION STABILITY
# ==============================================================================
def report_iteration_stability(df: pd.DataFrame):
    if "Iteration" not in df.columns or df["Iteration"].nunique() < 2:
        return

    print(f"\n{DIVIDER}")
    print(h("  ITERATION STABILITY"))
    print(DIVIDER)

    iters = sorted(df["Iteration"].unique())
    print(f"  Iterations found: {iters}")

    # count distinct features per iteration per brain
    stab = (df.groupby(["Brain","Iteration"])["Feature"]
             .nunique()
             .reset_index()
             .rename(columns={"Feature":"n_features"}))
    print()
    print(f"  {'Brain':<12} {'Iter':>5} {'N_Features':>12}")
    print("  " + "─" * 32)
    for _, r in stab.iterrows():
        print(f"  {r['Brain']:<12} {int(r['Iteration']):>5} {int(r['n_features']):>12}")

    # features that persisted all iterations (per brain)
    max_iter = df["Iteration"].max()
    all_iters_features = (df.groupby(["Brain","Feature"])["Iteration"]
                           .nunique()
                           .reset_index()
                           .query(f"Iteration == {max_iter}"))
    print()
    print(h(f"  Features present in ALL {int(max_iter)} iterations:"))
    for brain in df["Brain"].unique():
        sub = all_iters_features[all_iters_features["Brain"] == brain]
        print(f"\n  {brain}: {len(sub)} features")
        for f in sub["Feature"].tolist():
            print(f"    ✔ {f}")


# ==============================================================================
# RISK FLAGS
# ==============================================================================
def report_risk_flags(df: pd.DataFrame):
    print(f"\n{DIVIDER}")
    print(h("  RISK FLAGS & WARNINGS"))
    print(DIVIDER)

    flags = []

    # high correlation clusters
    if df["Max_R"].notna().any():
        high_corr = df[df["Max_R"] > 0.85]
        if not high_corr.empty:
            flags.append(warn(f"  ⚠  {len(high_corr)} row(s) with Max_R > 0.85 "
                              f"(risk of redundancy)"))

    # zero-importance features
    zero_imp = df[df["I_Norm"] == 0.0]["Feature"].nunique()
    if zero_imp:
        flags.append(warn(f"  ⚠  {zero_imp} feature(s) with I_Norm = 0.0"))

    # single-iteration features (low persistence)
    if "Persistence" in df.columns:
        low_pers = df[df["Persistence"] == 1]["Feature"].nunique()
        total    = df["Feature"].nunique()
        pct      = low_pers / total * 100 if total else 0
        if pct > 50:
            flags.append(warn(f"  ⚠  {low_pers}/{total} ({pct:.0f}%) features "
                              f"appeared in only 1 iteration — consider more iterations"))

    # lock features missing from CSV
    BRAIN_LOCKS = {
        "DIRECTION": ["LENS_90_cog_20_z_slope"],
        "EASE":      ["LENS_90_cog_20_z_sos"],
        "EXP":       ["LENS_10_hurst_50_z", "LENS_90_cog_20_z_sos"],
    }
    all_features = set(df["Feature"].unique())
    for brain, locks in BRAIN_LOCKS.items():
        for lk in locks:
            if lk not in all_features:
                flags.append(bad(f"  ✖  BRAIN_LOCK '{lk}' ({brain}) not found in CSV"))

    # WIN rolling group — expect all 3 windows (factory uses 10/30/60)
    expected_win = {"WIN_10_cog_20_pct", "WIN_30_cog_20_pct", "WIN_60_cog_20_pct"}
    missing_win  = expected_win - all_features
    if missing_win:
        flags.append(warn(f"  ⚠  WIN rolling COG features missing: "
                          f"{', '.join(sorted(missing_win))}"))

    # log-ratio families — check at least one ratio feature made it through
    ratio_families = {
        "kalman_sma_ratio", "tema_kalman_ratio", "exhaustion",
        "ratio_acc", "ratio_snr", "curvature_diff", "cycle_vs_trend",
        "ratio_eff_slope", "ratio_pers_slope", "ratio_struct",
        "adx_entropy_ratio", "ratio_breakout_eff",
    }
    found_ratio_fams = {
        parse_feature(f)["family"]
        for f in all_features
        if parse_feature(f)["family"] in ratio_families
    }
    if not found_ratio_fams:
        flags.append(warn("  ⚠  No Log-Ratio Protocol features found — "
                          "check factory ran v3.19.18+"))

    # LENS parity — every LENS_10 feature should have a LENS_90 counterpart
    l10 = {f for f in all_features if f.startswith("LENS_10_")}
    l90 = {f for f in all_features if f.startswith("LENS_90_")}
    def _strip_lens(name):
        return re.sub(r"^LENS_\d+_", "", name)
    l10_bodies = {_strip_lens(f) for f in l10}
    l90_bodies = {_strip_lens(f) for f in l90}
    only_10 = l10_bodies - l90_bodies
    only_90 = l90_bodies - l10_bodies
    if only_10:
        flags.append(warn(f"  ⚠  {len(only_10)} feature(s) have LENS_10 but no LENS_90 "
                          f"counterpart: {', '.join(sorted(only_10)[:5])}..."))
    if only_90:
        flags.append(warn(f"  ⚠  {len(only_90)} feature(s) have LENS_90 but no LENS_10 "
                          f"counterpart: {', '.join(sorted(only_90)[:5])}..."))

    # Unknown families (not in any known indicator group)
    known_fams = {f for fams in INDICATOR_GROUPS.values() for f in fams}
    unknown_fams = set(df["family"].dropna().unique()) - known_fams
    if unknown_fams:
        flags.append(warn(f"  ⚠  Unknown indicator families (not in INDICATOR_GROUPS): "
                          f"{', '.join(sorted(unknown_fams))}"))

    if flags:
        for f in flags:
            print(f)
    else:
        print(ok("  No risk flags detected."))


# ==============================================================================
# VITALITY GATE
# ==============================================================================
def report_vitality_gate(df: pd.DataFrame, num_iters: int):
    """
    Best Practice #1: Audit Vitality Gate
    ─ Mean I_Norm of top-19 per brain must be > 0.05  (signal floor)
    ─ Mean I_Norm > 0.90 triggers a leakage warning
    ─ Alpha Pillars: features surviving ≥ 80% of iterations
    """
    print(f"\n{DIVIDER}")
    print(h("  AUDIT VITALITY GATE  (Best Practice #1 & #2)"))
    print(DIVIDER)

    alpha_threshold = 0.80 * num_iters   # 80% persistence floor for Alpha Pillars

    for brain in sorted(df["Brain"].unique()):
        bdf = df[df["Brain"] == brain]
        top = _top_features(bdf, n=19, num_iters=num_iters)
        mean_imp = top["Avg_INorm"].mean()
        max_imp  = top["Avg_INorm"].max()

        if mean_imp < 0.05:
            gate_icon = bad("✖  FAIL")
            gate_note = "Model found no signal — feature seeds are noisy for this task."
        elif mean_imp > 0.90:
            gate_icon = warn("⚠  LEAKAGE?")
            gate_note = "Signals this powerful rarely exist in non-lagged price data."
        else:
            gate_icon = ok("✔  PASS")
            gate_note = "Signal floor healthy."

        alpha_pillars = top[top["Persistence"] >= alpha_threshold]
        model_note = ("GRU — target Directional Hit Rate 52–60%"
                      if brain == "DIRECTION"
                      else "LSTM — target MAE ≈ 0.014")

        print(f"\n  {h(brain)}  [{model_note}]")
        print(f"    Mean I_Norm (top-19) : {mean_imp:.4f}   Max: {max_imp:.4f}   "
              f"→ {gate_icon}  {gate_note}")
        print(f"    Alpha Pillars (≥{alpha_threshold:.0f}/{num_iters} iters): "
              f"{len(alpha_pillars)} features")
        for _, r in alpha_pillars.iterrows():
            pct = r["Persistence"] / num_iters * 100
            print(f"      ⭐  {r['Feature']:<42} persist={r['Persistence']:.0f}/{num_iters}"
                  f" ({pct:.0f}%)  imp={r['Avg_INorm']:.4f}")


# ==============================================================================
# FINAL SOVEREIGN SCORING TABLE  (70/15/15)
# ==============================================================================
def _redundancy_cull(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Best Practice #4: Redundancy Cull
    For each indicator family, only the window (LENS_10 vs LENS_90 etc.)
    with the highest A_Impact survives. Lower-impact duplicates are flagged.
    Transforms (_z, _z_slope, _z_sos) of the winning window are preserved.
    """
    # Extract family and window from feature names using the existing parser
    parsed = grp["Feature"].apply(parse_feature).apply(pd.Series)
    grp = grp.copy()
    grp["_family"]  = parsed["family"].values
    grp["_lens"]    = parsed["lens"].values
    grp["_window"]  = parsed["window"].values
    grp["_suffix"]  = parsed["suffix"].values

    # For LENS features: per family, find the window with highest avg A_Impact
    lens_mask = grp["_lens"].notna()
    culled_flags = pd.Series(False, index=grp.index)

    if lens_mask.any():
        lens_grp = grp[lens_mask].copy()
        # best window per family = the lens value with highest avg Avg_INorm
        best_window = (lens_grp.groupby("_family")
                       .apply(lambda x: x.groupby("_lens")["Avg_INorm"].mean().idxmax())
                       .to_dict())
        # flag any row whose lens is NOT the best for its family
        def _is_culled(row):
            fam = row["_family"]
            if fam in best_window and pd.notna(row["_lens"]):
                return int(row["_lens"]) != best_window[fam]
            return False
        for idx, row in lens_grp.iterrows():
            culled_flags.at[idx] = _is_culled(row)

    grp["_culled"] = culled_flags
    return grp


def _physics_suite(suffix: str) -> str:
    return PHYSICS_SUITE.get(str(suffix).lower(), "Other")


def report_sovereign_summary(df: pd.DataFrame, num_iters: int):
    """
    Best Practices #3 + #4 + #5 combined:
    ─ 70/15/15 Sovereign Score
    ─ Redundancy Cull (flag duplicate-window same-family features)
    ─ Physics Suite categorisation (Position / Velocity / Acceleration)
    ─ Family diversity health check (≥5 unique seeds required)
    ─ Brain-specific performance notes
    """
    print(f"\n{DIVIDER}")
    print(h("  FINAL SOVEREIGN SCORING TABLE  (70% Impact · 15% Persist · 15% UV%)"))
    print(DIVIDER)
    print(f"  Scoring formula: Sov = 0.70×A_Impact_norm + 0.15×Persist_frac + 0.15×UV_norm")
    print(f"  Redundancy cull: per family → keep window with highest A_Impact only")
    print()

    all_final = []

    for brain in sorted(df["Brain"].unique()):
        bdf = df[df["Brain"] == brain]

        # aggregate
        grp = (bdf.groupby("Feature")
               .agg(
                   Persistence = ("Persistence", "first"),
                   Avg_INorm   = ("I_Norm",       "mean"),
                   Avg_UV      = ("UV%",           "mean"),
                   Max_R_avg   = ("Max_R",         "mean"),
                   Is_Locked   = ("Is_Locked",     "first"),
               )
               .reset_index())
        grp = _compute_sovereign_score(grp, num_iters)
        grp = _redundancy_cull(grp)
        grp["Brain"] = brain

        # sort by score descending, culled features pushed to bottom
        grp = grp.sort_values(["_culled", "Sov_Score"], ascending=[True, False])
        top = grp.head(19).reset_index(drop=True)
        all_final.append(top)

        alpha_thresh = 0.80 * num_iters
        brain_target = ("GRU → Hit Rate 52-60%"
                        if brain == "DIRECTION"
                        else "LSTM → MAE ≈ 0.014")

        # unique seed families in top-19
        parsed_top = top["Feature"].apply(parse_feature).apply(pd.Series)
        unique_seeds = parsed_top["family"].nunique()
        diversity_ok = unique_seeds >= 5
        diversity_icon = ok(f"✔ {unique_seeds} seeds") if diversity_ok else warn(f"⚠ {unique_seeds} seeds (<5)")

        print(f"\n  {'─'*74}")
        print(h(f"  BRAIN: {brain}   [{brain_target}]   Family diversity: {diversity_icon}"))
        print(f"  {'─'*74}")
        print(f"  {'RNK':<4} {'FEATURE':<42} {'SUITE':<22} {'PERS%':>6} "
              f"{'IMP':>6} {'UV%':>6} {'SOV':>7} {'FLAG'}")
        print(f"  {'─'*100}")

        suite_counts = {"Position / Z": 0, "Velocity / Slope": 0,
                        "Acceleration / SOS": 0, "WIN Rolling %": 0}

        for i, row in top.iterrows():
            lock_icon  = "🔒" if row["Is_Locked"] else "  "
            cull_icon  = warn(" ✂CULL") if row["_culled"] else ""
            alpha_icon = " ⭐" if row["Persistence"] >= alpha_thresh else ""
            persist_pct = row["Persistence"] / num_iters * 100
            suite      = _physics_suite(row["_suffix"])
            uv_s       = f"{row['Avg_UV']:>5.1f}" if pd.notna(row["Avg_UV"]) else "  —  "

            if suite in suite_counts:
                suite_counts[suite] += 1

            flag_str = f"{lock_icon}{alpha_icon}{cull_icon}"
            print(f"  {i+1:02d}.  {row['Feature']:<42} {suite:<22} "
                  f"{persist_pct:>5.0f}%  {row['Avg_INorm']:>6.4f}  "
                  f"{uv_s}  {row['Sov_Score']:>7.4f}  {flag_str}")

        # Physics Suite summary
        print(f"\n  Physics Suite breakdown:")
        for suite_name, cnt in suite_counts.items():
            bar = "█" * cnt
            print(f"    {suite_name:<22} {cnt:>2}  {bar}")

        # Unique family list
        print(f"\n  Unique indicator seeds in final-19:")
        for fam in sorted(parsed_top["family"].unique()):
            grp_label = _FAM_TO_GROUP.get(fam, "Other")
            print(f"    {fam:<30} [{grp_label}]")

    return all_final


# ==============================================================================
# MATPLOTLIB CHARTS (optional)
# ==============================================================================
def save_charts(df: pd.DataFrame, out_dir: str, num_iters: int):
    if not HAS_MPL:
        print(warn("  matplotlib not available — skipping charts."))
        return

    os.makedirs(out_dir, exist_ok=True)
    brains = sorted(df["Brain"].unique())
    n = len(brains)

    # ── Chart 1: Top-10 Sovereign Score per brain ─────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), squeeze=False)
    for ax, brain in zip(axes[0], brains):
        bdf  = df[df["Brain"] == brain]
        top  = _top_features(bdf, n=10, num_iters=num_iters)
        bars = ax.barh(top["Feature"][::-1], top["Sov_Score"][::-1],
                       color=["#e63946" if b else "#457b9d"
                               for b in top["Is_Locked"][::-1]])
        ax.set_title(f"{brain} — Top 10 Sovereign Score", fontweight="bold")
        ax.set_xlabel("Sovereign Score (70% Impact · 15% Persist · 15% UV%)")
        ax.tick_params(axis="y", labelsize=7)
        # legend patches
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#e63946", label="Locked"),
                            Patch(color="#457b9d", label="Selected")],
                  loc="lower right", fontsize=7)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "sovereign_top10.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(ok(f"  Chart saved: {p1}"))

    # ── Chart 2: LENS split stacked bar ───────────────────────────────────────
    lens_pivot = (df.groupby(["Brain","src_type"])["I_Norm"]
                    .mean()
                    .unstack("src_type", fill_value=0))
    fig2, ax2 = plt.subplots(figsize=(max(6, n * 2), 4))
    lens_pivot.plot(kind="bar", ax=ax2, colormap="Set2")
    ax2.set_title("Avg I_Norm by LENS type per Brain", fontweight="bold")
    ax2.set_ylabel("Avg I_Norm")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=0)
    ax2.legend(title="LENS", fontsize=8)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "lens_split.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(ok(f"  Chart saved: {p2}"))

    # ── Chart 3: Suffix distribution ──────────────────────────────────────────
    if df["suffix"].notna().any():
        suf_pivot = (df.groupby(["Brain","suffix"])["I_Norm"]
                       .mean()
                       .unstack("suffix", fill_value=0))
        fig3, ax3 = plt.subplots(figsize=(max(6, n * 2), 4))
        suf_pivot.plot(kind="bar", ax=ax3, colormap="tab10")
        ax3.set_title("Avg I_Norm by Suffix (z / z_slope / z_sos)", fontweight="bold")
        ax3.set_ylabel("Avg I_Norm")
        ax3.set_xlabel("")
        ax3.tick_params(axis="x", rotation=0)
        ax3.legend(title="Suffix", fontsize=8)
        fig3.tight_layout()
        p3 = os.path.join(out_dir, "suffix_split.png")
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        print(ok(f"  Chart saved: {p3}"))

    # ── Chart 4: Impact distribution histogram ────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    for brain in brains:
        sub = df[df["Brain"] == brain]["I_Norm"]
        ax4.hist(sub, bins=30, alpha=0.55, label=brain)
    ax4.set_title("I_Norm distribution per Brain", fontweight="bold")
    ax4.set_xlabel("I_Norm")
    ax4.set_ylabel("Count")
    ax4.legend()
    fig4.tight_layout()
    p4 = os.path.join(out_dir, "inorm_dist.png")
    fig4.savefig(p4, dpi=150)
    plt.close(fig4)
    print(ok(f"  Chart saved: {p4}"))

    # ── Chart 5: Indicator group importance per brain ─────────────────────────
    df5 = df.copy()
    df5["group"] = df5["family"].map(_FAM_TO_GROUP).fillna("Other")
    grp_pivot = (df5.groupby(["Brain", "group"])["I_Norm"]
                     .mean()
                     .unstack("group", fill_value=0))
    fig5, ax5 = plt.subplots(figsize=(max(8, n * 3), 5))
    grp_pivot.plot(kind="bar", ax=ax5, colormap="Paired", width=0.7)
    ax5.set_title("Avg I_Norm by Indicator Group per Brain", fontweight="bold")
    ax5.set_ylabel("Avg I_Norm")
    ax5.set_xlabel("")
    ax5.tick_params(axis="x", rotation=0)
    ax5.legend(title="Indicator Group", fontsize=8, loc="upper right")
    fig5.tight_layout()
    p5 = os.path.join(out_dir, "indicator_groups.png")
    fig5.savefig(p5, dpi=150)
    plt.close(fig5)
    print(ok(f"  Chart saved: {p5}"))

    # ── Chart 6: LENS_10 vs LENS_90 avg importance per brain ─────────────────
    lens_df = df[df["lens"].notna()]
    if not lens_df.empty:
        lens_piv = (lens_df.groupby(["Brain", "src_type"])["I_Norm"]
                           .mean()
                           .unstack("src_type", fill_value=0))
        fig6, ax6 = plt.subplots(figsize=(max(6, n * 2), 4))
        lens_piv.plot(kind="bar", ax=ax6, colormap="coolwarm", width=0.6)
        ax6.set_title("LENS_10 vs LENS_90 — Avg I_Norm per Brain", fontweight="bold")
        ax6.set_ylabel("Avg I_Norm")
        ax6.set_xlabel("")
        ax6.tick_params(axis="x", rotation=0)
        ax6.legend(title="LENS", fontsize=8)
        fig6.tight_layout()
        p6 = os.path.join(out_dir, "lens_10_vs_90.png")
        fig6.savefig(p6, dpi=150)
        plt.close(fig6)
        print(ok(f"  Chart saved: {p6}"))


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Assess a Sovereign_Audit_Master.csv produced by "
                    "sovereign_titan_v3_19_18.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python sovereign_audit_assessor.py
              python sovereign_audit_assessor.py path/to/Sovereign_Audit_Master_20240101_120000.csv
              python sovereign_audit_assessor.py audit.csv --charts --out_dir ./audit_charts
        """))
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path to Sovereign_Audit_Master CSV (auto-detect latest if omitted)")
    parser.add_argument("--charts", action="store_true",
                        help="Save matplotlib charts to --out_dir")
    parser.add_argument("--out_dir", default="./sovereign_charts",
                        help="Directory for chart output (default: ./sovereign_charts)")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override total iteration count used for persistence display")
    args = parser.parse_args()

    # ── locate CSV ────────────────────────────────────────────────────────────
    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(
            glob.glob("**/Sovereign_Audit_Master*.csv", recursive=True),
            key=os.path.getmtime, reverse=True)
        if not candidates:
            sys.exit(bad("No Sovereign_Audit_Master*.csv found. Pass the path explicitly."))
        csv_path = candidates[0]
        print(warn(f"  Auto-detected: {csv_path}"))

    if not os.path.isfile(csv_path):
        sys.exit(bad(f"File not found: {csv_path}"))

    df = load_csv(csv_path)
    brains     = sorted(df["Brain"].unique())
    num_iters  = args.iters or (int(df["Iteration"].max()) if "Iteration" in df.columns
                                else df.groupby(["Brain","Feature"]).size().max())

    print(f"\n{DIVIDER}")
    print(h(f"  SOVEREIGN AUDIT ASSESSOR"))
    print(f"  File     : {os.path.basename(csv_path)}")
    print(f"  Rows     : {len(df):,}")
    print(f"  Brains   : {', '.join(brains)}")
    print(f"  Iters    : {num_iters}")
    print(f"  Features : {df['Feature'].nunique():,} unique")
    print(DIVIDER)

    # ── vitality gate ─────────────────────────────────────────────────────────
    report_vitality_gate(df, num_iters)

    # ── per-brain detail ──────────────────────────────────────────────────────
    for brain in brains:
        bdf = df[df["Brain"] == brain]
        report_brain(brain, bdf, num_iters)

    # ── final sovereign scoring table (70/15/15) ──────────────────────────────
    report_sovereign_summary(df, num_iters)

    # ── input group analysis ──────────────────────────────────────────────────
    report_input_groups(df)

    # ── cross-brain ───────────────────────────────────────────────────────────
    if len(brains) > 1:
        report_cross_brain(df, brains)

    # ── iteration stability ───────────────────────────────────────────────────
    report_iteration_stability(df)

    # ── risk flags ────────────────────────────────────────────────────────────
    report_risk_flags(df)

    # ── charts ────────────────────────────────────────────────────────────────
    if args.charts:
        print(f"\n{DIVIDER}")
        print(h("  GENERATING CHARTS"))
        print(DIVIDER)
        save_charts(df, args.out_dir, num_iters)

    print(f"\n{DIVIDER}")
    print(ok("  Assessment complete."))
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()
