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
# FEATURE NAME PARSER
# ==============================================================================
_LENS_RE    = re.compile(r"^LENS_(\d+)_", re.IGNORECASE)
_WIN_RE     = re.compile(r"^WIN_(\d+)_",  re.IGNORECASE)
_SUFFIX_RE  = re.compile(r"_(z(?:_slope|_sos)?)$", re.IGNORECASE)

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
    df["src_type"]= df["lens"].apply(lambda v: f"LENS_{v}" if pd.notna(v) else "WIN")

    return df


# ==============================================================================
# PER-BRAIN REPORT
# ==============================================================================
def _top_features(brain_df: pd.DataFrame, n: int = 19) -> pd.DataFrame:
    """Aggregate by Feature and rank."""
    grp = (brain_df.groupby("Feature")
           .agg(
               Runs       = ("I_raw", "count"),
               Avg_IRaw   = ("I_raw", "mean"),
               Avg_INorm  = ("I_Norm", "mean"),
               Avg_UV     = ("UV%", "mean"),
               Max_R_avg  = ("Max_R", "mean"),
               Persistence= ("Persistence", "first"),
               Is_Locked  = ("Is_Locked", "first"),
               Model_Type = ("Model_Type", "first"),
           )
           .reset_index())
    # sovereign score: persistence × avg normalised importance
    grp["Sov_Score"] = grp["Persistence"] * grp["Avg_INorm"]
    return (grp.sort_values("Sov_Score", ascending=False)
               .head(n)
               .reset_index(drop=True))


def report_brain(brain: str, bdf: pd.DataFrame, num_iters: int):
    top = _top_features(bdf)
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
           f"{'AVG_IMP':>8} {'SOV_SCR':>8} {'LK':>4}")
    print(hdr)
    print("  " + "─" * 68)

    for i, row in top.iterrows():
        lock_icon = "🔒" if row["Is_Locked"] else "  "
        persist_s = f"{int(row['Persistence'])}/{num_iters}"
        color     = GREEN if row["Is_Locked"] else RESET
        line = (f"  {i+1:02d}.  {color}{row['Feature']:<40}{RESET} "
                f"{persist_s:>7}  {row['Avg_INorm']:>8.4f}  "
                f"{row['Sov_Score']:>8.4f}  {lock_icon}")
        print(line)

    print()

    # ── family breakdown ──────────────────────────────────────────────────────
    fam_counts = bdf.groupby("family")["I_Norm"].mean().sort_values(ascending=False)
    print(h("  Family breakdown (avg I_Norm):"))
    for fam, val in fam_counts.head(10).items():
        bar = "█" * int(val * 40)
        print(f"    {fam:<28} {val:.4f}  {bar}")

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

    if flags:
        for f in flags:
            print(f)
    else:
        print(ok("  No risk flags detected."))


# ==============================================================================
# SOVEREIGN SCORE SUMMARY TABLE
# ==============================================================================
def report_sovereign_summary(df: pd.DataFrame, num_iters: int):
    print(f"\n{DIVIDER}")
    print(h("  SOVEREIGN SCORE SUMMARY (all brains, top 25)"))
    print(DIVIDER)

    grp = (df.groupby(["Brain","Feature"])
             .agg(
                 Persistence = ("Persistence","first"),
                 Avg_INorm   = ("I_Norm","mean"),
                 Avg_UV      = ("UV%","mean"),
             )
             .reset_index())
    grp["Sov_Score"] = grp["Persistence"] * grp["Avg_INorm"]
    top = grp.sort_values("Sov_Score", ascending=False).head(25).reset_index(drop=True)

    print(f"\n  {'RNK':<4} {'BRAIN':<12} {'FEATURE':<40} {'PERSIST':>7} "
          f"{'AVG_IMP':>8} {'SOV_SCR':>8}")
    print("  " + "─" * 82)
    for i, row in top.iterrows():
        persist_s = f"{int(row['Persistence'])}/{num_iters}"
        print(f"  {i+1:02d}.  {row['Brain']:<12} {row['Feature']:<40} "
              f"{persist_s:>7}  {row['Avg_INorm']:>8.4f}  {row['Sov_Score']:>8.4f}")


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
        top  = _top_features(bdf, n=10)
        bars = ax.barh(top["Feature"][::-1], top["Sov_Score"][::-1],
                       color=["#e63946" if b else "#457b9d"
                               for b in top["Is_Locked"][::-1]])
        ax.set_title(f"{brain} — Top 10 Sovereign Score", fontweight="bold")
        ax.set_xlabel("Sovereign Score (Persistence × Avg I_Norm)")
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

    # ── per-brain detail ──────────────────────────────────────────────────────
    for brain in brains:
        bdf = df[df["Brain"] == brain]
        report_brain(brain, bdf, num_iters)

    # ── global sovereign summary ──────────────────────────────────────────────
    report_sovereign_summary(df, num_iters)

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
