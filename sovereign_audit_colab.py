# ==============================================================================
# SOVEREIGN AUDIT ASSESSOR — GOOGLE COLAB VERSION
# Run each cell in order. Cell boundaries marked with  # %%
# ==============================================================================

# %% [1] INSTALL & MOUNT DRIVE
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# %% [2] IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, kendalltau
from IPython.display import display, HTML

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})

# %% [3] LOCATE FILE — edit SEARCH_ROOT if your Drive folder differs
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_ROOT = "/content/drive/MyDrive"          # ← change if needed
N_FEATURES  = 19                                 # slots per brain

# Auto-detect the most recent Sovereign_Audit_Master CSV
candidates = sorted(
    glob.glob(f"{SEARCH_ROOT}/**/Sovereign_Audit_Master*.csv", recursive=True),
    key=os.path.getmtime, reverse=True
)
if not candidates:
    raise FileNotFoundError(
        "No Sovereign_Audit_Master*.csv found under "
        f"{SEARCH_ROOT}\nCheck your Drive folder or set SEARCH_ROOT."
    )

CSV_PATH = candidates[0]
print(f"✅ Found: {CSV_PATH}")

# %% [4] LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"Feature", "Brain", "Iteration", "I_Norm"}
OPTIONAL_DEFAULTS = {
    "I_raw"      : 0.0,
    "I_stability": 0.0,
    "UV_score"   : np.nan,
    "UV%"        : np.nan,
    "Stab_Norm"  : np.nan,
    "Sov_Score"  : np.nan,
    "Max_R"      : np.nan,
    "Is_Locked"  : False,
    "Persistence": 1,
    "A_Impact"   : np.nan,
    "A_UV"       : np.nan,
    "Model_Type" : "LSTM",
    "Timestamp"  : "",
}

df = pd.read_csv(CSV_PATH)
missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

for col, default in OPTIONAL_DEFAULTS.items():
    if col not in df.columns:
        df[col] = default

# normalise types
df["Iteration"]  = pd.to_numeric(df["Iteration"],  errors="coerce").fillna(1).astype(int)
df["Is_Locked"]  = df["Is_Locked"].astype(str).str.lower().isin(["true","1","yes"])
df["I_Norm"]     = pd.to_numeric(df["I_Norm"],     errors="coerce").fillna(0.0)
df["Sov_Score"]  = pd.to_numeric(df["Sov_Score"],  errors="coerce")
df["UV%"]        = pd.to_numeric(df["UV%"],         errors="coerce")
df["A_Impact"]   = pd.to_numeric(df["A_Impact"],   errors="coerce")
df["A_UV"]       = pd.to_numeric(df["A_UV"],       errors="coerce")

BRAINS    = sorted(df["Brain"].unique())
ITERS     = sorted(df["Iteration"].unique())
N_ITERS   = len(ITERS)

print(f"\n{'─'*60}")
print(f"  Rows       : {len(df):,}")
print(f"  Brains     : {BRAINS}")
print(f"  Iterations : {ITERS}  (n={N_ITERS})")
print(f"  Unique feat: {df['Feature'].nunique():,}")
print(f"{'─'*60}")

# %% [5] HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def rank_iter(brain_df: pd.DataFrame, iteration: int) -> pd.Series:
    """Return {feature: rank} for one iteration of one brain (lower = better)."""
    sub = brain_df[brain_df["Iteration"] == iteration].copy()
    # prefer Sov_Score if available, else I_Norm
    score_col = "Sov_Score" if sub["Sov_Score"].notna().any() else "I_Norm"
    sub = sub.sort_values(score_col, ascending=False).reset_index(drop=True)
    return pd.Series(sub.index.values + 1, index=sub["Feature"].values)


def iter_rank_corr(brain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman rank correlation between every pair of consecutive
    iterations for a single brain.
    Returns a DataFrame with columns [iter_a, iter_b, spearman_r, kendall_tau,
    top19_overlap, top19_new_entries].
    """
    rows = []
    all_feats = sorted(brain_df["Feature"].unique())
    for i in range(len(ITERS) - 1):
        ia, ib = ITERS[i], ITERS[i + 1]
        ra = rank_iter(brain_df, ia).reindex(all_feats).fillna(len(all_feats) + 1)
        rb = rank_iter(brain_df, ib).reindex(all_feats).fillna(len(all_feats) + 1)

        sp, _  = spearmanr(ra, rb)
        kt, _  = kendalltau(ra, rb)

        # top-N overlap
        top_a = set(ra.nsmallest(N_FEATURES).index)
        top_b = set(rb.nsmallest(N_FEATURES).index)
        overlap = len(top_a & top_b)
        new_in  = len(top_b - top_a)

        rows.append(dict(iter_a=ia, iter_b=ib,
                         spearman_r=round(sp, 4),
                         kendall_tau=round(kt, 4),
                         top19_overlap=overlap,
                         top19_new_entries=new_in))
    return pd.DataFrame(rows)


def sov_score_cv(brain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Coefficient of variation of Sov_Score (or I_Norm) per feature across
    iterations — measures how stable each feature's score is.
    """
    score_col = "Sov_Score" if brain_df["Sov_Score"].notna().any() else "I_Norm"
    grp = (brain_df.groupby("Feature")[score_col]
           .agg(["mean","std","count"])
           .rename(columns={"mean":"avg","std":"sd","count":"n_obs"}))
    grp["cv"] = (grp["sd"] / grp["avg"].replace(0, np.nan)).fillna(0)
    return grp.sort_values("avg", ascending=False)


def enough_iterations_verdict(corr_df: pd.DataFrame, cv_df: pd.DataFrame,
                               brain: str) -> dict:
    """
    Produce a structured verdict.
    Criteria (all must pass for SUFFICIENT):
      A. Last rank correlation  ρ ≥ 0.85
      B. Last top-19 overlap   ≥ 16/19  (≥ 84%)
      C. Median feature CV     ≤ 0.20
    """
    if corr_df.empty:
        return {"verdict": "CANNOT_ASSESS", "reason": "Only 1 iteration found."}

    last = corr_df.iloc[-1]
    rho        = last["spearman_r"]
    overlap    = last["top19_overlap"]
    median_cv  = cv_df["cv"].median()

    pass_rho     = rho >= 0.85
    pass_overlap = overlap >= 16
    pass_cv      = median_cv <= 0.20

    all_pass = pass_rho and pass_overlap and pass_cv

    notes = []
    if not pass_rho:
        notes.append(f"rank ρ={rho:.3f} < 0.85 — rankings still shifting")
    if not pass_overlap:
        notes.append(f"top-19 overlap={overlap}/19 < 16 — slot instability")
    if not pass_cv:
        notes.append(f"median CV={median_cv:.3f} > 0.20 — score variance too high")

    # trend: is ρ improving?
    trending_up = (len(corr_df) >= 2 and
                   corr_df["spearman_r"].iloc[-1] > corr_df["spearman_r"].iloc[-2])

    return {
        "verdict"     : "✅ SUFFICIENT" if all_pass else "⚠️  MORE NEEDED",
        "brain"       : brain,
        "n_iters"     : N_ITERS,
        "last_rho"    : rho,
        "last_overlap": overlap,
        "median_cv"   : median_cv,
        "trending_up" : trending_up,
        "pass_rho"    : pass_rho,
        "pass_overlap": pass_overlap,
        "pass_cv"     : pass_cv,
        "notes"       : notes,
        "recommend_n" : max(N_ITERS + 3, 8) if not all_pass else N_ITERS,
    }


# %% [6] CONVERGENCE ANALYSIS PER BRAIN
# ─────────────────────────────────────────────────────────────────────────────
verdicts   = {}
corr_tables = {}
cv_tables   = {}

for brain in BRAINS:
    bdf   = df[df["Brain"] == brain]
    corr  = iter_rank_corr(bdf)
    cv    = sov_score_cv(bdf)
    verd  = enough_iterations_verdict(corr, cv, brain)

    verdicts[brain]    = verd
    corr_tables[brain] = corr
    cv_tables[brain]   = cv

# ── Print convergence tables ──────────────────────────────────────────────────
print("\n" + "═"*70)
print("  CONVERGENCE ANALYSIS")
print("═"*70)

for brain in BRAINS:
    corr = corr_tables[brain]
    verd = verdicts[brain]
    print(f"\n  ┌─ {brain} {'─'*(54-len(brain))}┐")
    if corr.empty:
        print(f"  │  Only 1 iteration — cannot compute convergence.        │")
    else:
        print(f"  │  {'Iters':>8}  {'ρ Spearman':>11}  {'τ Kendall':>10}  "
              f"{'Overlap/19':>10}  {'New slots':>9}  │")
        print(f"  │  {'─'*8}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*9}  │")
        for _, r in corr.iterrows():
            flag = " ✓" if r["spearman_r"] >= 0.85 else " ⚠"
            print(f"  │  {int(r.iter_a):>3}→{int(r.iter_b):<4}  "
                  f"{r.spearman_r:>10.4f}  {r.kendall_tau:>10.4f}  "
                  f"{int(r.top19_overlap):>8}/19  {int(r.top19_new_entries):>9}  │{flag}")
    print(f"  └{'─'*57}┘")
    print(f"\n  VERDICT [{brain}]: {verd['verdict']}")
    print(f"    last ρ={verd['last_rho']:.4f}  overlap={verd['last_overlap']}/19"
          f"  median CV={verd['median_cv']:.4f}"
          f"  trending={'↑' if verd['trending_up'] else '─'}")
    if verd["notes"]:
        for n in verd["notes"]:
            print(f"    ⚠  {n}")
    if verd["verdict"] != "✅ SUFFICIENT":
        print(f"    → Recommended: run at least {verd['recommend_n']} total iterations")


# %% [7] FINAL 19-FEATURE ROSTER PER BRAIN
# ─────────────────────────────────────────────────────────────────────────────
def build_final_roster(brain_df: pd.DataFrame, n: int = N_FEATURES) -> pd.DataFrame:
    """
    Aggregate all iterations into a single ranked list.
    Score hierarchy:
      1. A_Impact   — pre-aggregated cross-iteration impact (if present)
      2. Sov_Score  — per-row sovereign score (averaged)
      3. I_Norm     — raw normalised importance (fallback)
    Also computes:
      - Persistence (fraction of iterations the feature appeared)
      - Stability (mean Stab_Norm across iterations)
      - UV% (mean UV%)
      - Is_Locked flag
    """
    use_a_impact = brain_df["A_Impact"].notna().any()
    score_col    = "A_Impact" if use_a_impact else (
                   "Sov_Score" if brain_df["Sov_Score"].notna().any() else "I_Norm")

    grp = (brain_df.groupby("Feature")
           .agg(
               Runs       = ("Iteration",  "nunique"),
               Score      = (score_col,    "mean"),
               Avg_INorm  = ("I_Norm",     "mean"),
               Avg_Stab   = ("Stab_Norm",  "mean"),
               Avg_UV     = ("UV%",        "mean"),
               A_UV       = ("A_UV",       "first"),
               Max_R      = ("Max_R",      "mean"),
               Is_Locked  = ("Is_Locked",  "first"),
               Model_Type = ("Model_Type", "first"),
           )
           .reset_index())

    grp["Persistence_pct"] = grp["Runs"] / N_ITERS * 100

    # locked features always included regardless of score
    locked  = grp[grp["Is_Locked"]].copy()
    unlocked = grp[~grp["Is_Locked"]].copy()
    unlocked = unlocked.sort_values("Score", ascending=False)

    slots_left = n - len(locked)
    final = pd.concat([locked, unlocked.head(slots_left)], ignore_index=True)
    final = final.sort_values("Score", ascending=False).reset_index(drop=True)
    final.index = final.index + 1   # 1-based rank
    return final, score_col


final_rosters = {}
for brain in BRAINS:
    bdf = df[df["Brain"] == brain]
    roster, score_col = build_final_roster(bdf)
    final_rosters[brain] = roster

    print(f"\n{'═'*78}")
    print(f"  FINAL {N_FEATURES}-FEATURE ROSTER ─ {brain}  "
          f"(scored by {score_col})")
    print(f"{'═'*78}")
    print(f"  {'RNK':<4} {'FEATURE':<40} {'SCORE':>7} {'IMP':>7} "
          f"{'STAB':>6} {'UV%':>6} {'PERS':>6} {'LK':>3}")
    print(f"  {'─'*72}")

    for rank, row in roster.iterrows():
        lock_s = "🔒" if row["Is_Locked"] else "  "
        stab_s = f"{row['Avg_Stab']:.3f}" if pd.notna(row['Avg_Stab']) else "  n/a"
        uv_s   = f"{row['Avg_UV']:.1f}" if pd.notna(row['Avg_UV']) else " n/a"
        pers_s = f"{row['Persistence_pct']:.0f}%"
        print(f"  {rank:02d}.  {row['Feature']:<40} {row['Score']:>7.4f} "
              f"{row['Avg_INorm']:>7.4f} {stab_s:>6} {uv_s:>6} "
              f"{pers_s:>5} {lock_s}")

print()


# %% [8] CHARTS
# ─────────────────────────────────────────────────────────────────────────────
n_b = len(BRAINS)

# ── 8A: Convergence curves ────────────────────────────────────────────────────
if N_ITERS > 1:
    fig, axes = plt.subplots(1, n_b, figsize=(6 * n_b, 4), squeeze=False)
    for ax, brain in zip(axes[0], BRAINS):
        corr = corr_tables[brain]
        if corr.empty:
            ax.text(0.5, 0.5, "Only 1 iteration", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
        else:
            x = [f"{int(r.iter_a)}→{int(r.iter_b)}" for _, r in corr.iterrows()]
            ax.plot(x, corr["spearman_r"], "o-", color="#457b9d", label="Spearman ρ", lw=2)
            ax.plot(x, corr["top19_overlap"] / N_FEATURES, "s--",
                    color="#e63946", label=f"Top-{N_FEATURES} overlap", lw=1.5)
            ax.axhline(0.85, ls=":", color="gray", lw=1, label="ρ=0.85 threshold")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{brain} — Ranking Convergence", fontweight="bold")
            ax.set_ylabel("Score / Overlap fraction")
            ax.set_xlabel("Iteration transition")
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Sovereign Feature Convergence Across Iterations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()

# ── 8B: Final roster — horizontal bar chart per brain ─────────────────────────
fig2, axes2 = plt.subplots(1, n_b, figsize=(9 * n_b, 7), squeeze=False)
for ax, brain in zip(axes2[0], BRAINS):
    roster = final_rosters[brain].reset_index(drop=False)   # reset so index is 0-based
    roster = roster.sort_values("Score", ascending=True)    # ascending for barh
    colors = ["#e63946" if lk else "#457b9d"
              for lk in roster["Is_Locked"]]
    bars = ax.barh(roster["Feature"], roster["Score"], color=colors)
    ax.set_xlabel("Score")
    ax.set_title(f"{brain} — Final {N_FEATURES} Features", fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e63946", label="Locked"),
                        Patch(color="#457b9d", label="Ranked")],
              loc="lower right", fontsize=8)

fig2.suptitle("Final Feature Rosters", fontsize=13, fontweight="bold", y=1.02)
fig2.tight_layout()
plt.show()

# ── 8C: Score CV heatmap (stability view) ─────────────────────────────────────
if N_ITERS > 1:
    # collect top-19 features per brain and their CVs
    all_top_feats = set()
    for brain in BRAINS:
        all_top_feats |= set(final_rosters[brain]["Feature"])

    cv_matrix = pd.DataFrame(index=sorted(all_top_feats), columns=BRAINS, dtype=float)
    for brain in BRAINS:
        cv = cv_tables[brain].reindex(sorted(all_top_feats))
        cv_matrix[brain] = cv["cv"]

    fig3, ax3 = plt.subplots(figsize=(max(5, 3 * n_b), max(8, len(all_top_feats) * 0.38)))
    im = ax3.imshow(cv_matrix.values.astype(float), aspect="auto",
                    cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax3.set_xticks(range(n_b))
    ax3.set_xticklabels(BRAINS, fontsize=9)
    ax3.set_yticks(range(len(cv_matrix)))
    ax3.set_yticklabels(cv_matrix.index, fontsize=6)
    ax3.set_title("Score CV per Feature per Brain\n(green=stable, red=volatile)",
                  fontweight="bold")
    plt.colorbar(im, ax=ax3, label="CV (σ/μ)")
    fig3.tight_layout()
    plt.show()


# %% [9] NEW TREND DIMENSIONS & RATIO FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# 4 missing trend dimensions  +  Top-10 high-value ratios for LSTM inputs.
# Requires a separate OHLCV CSV.  Set PRICE_CSV below, then run this cell.
# ─────────────────────────────────────────────────────────────────────────────
from scipy import signal as sp_signal
from scipy.stats import linregress as _lr

PRICE_CSV      = "/content/drive/MyDrive/price_data.csv"  # ← set your file
PRICE_DATE_COL = "Date"
PRICE_CLOSE_COL = "Close"
PRICE_HIGH_COL  = "High"
PRICE_LOW_COL   = "Low"

# ── Feature engineering helpers ───────────────────────────────────────────────
def _rolling_apply(series, window, func, raw=True):
    return series.rolling(window).apply(func, raw=raw)

def rolling_slope(s, w):
    def _s(y):
        x = np.arange(len(y))
        return _lr(x, y).slope
    return _rolling_apply(s, w, _s)

def rolling_r2(s, w):
    def _r(y):
        x = np.arange(len(y))
        return _lr(x, y).rvalue ** 2
    return _rolling_apply(s, w, _r)

def rolling_quad_curvature(s, w):
    """Trend Dim 2 — quadratic fit coefficient 'a' in ax²+bx+c."""
    def _q(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 2)[0]   # coefficient a
    return _rolling_apply(s, w, _q)

def rolling_spectral_power_ratio(s, w, low_frac=0.25):
    """Trend Dim 3 (Ehlers) — low-freq energy fraction; high → trend dominant."""
    def _spr(y):
        f, pxx = sp_signal.periodogram(y - y.mean())
        if pxx[1:].sum() == 0:
            return np.nan
        cut = max(1, int(len(f) * low_frac))
        return pxx[1:cut + 1].sum() / pxx[1:].sum()
    return _rolling_apply(s, w, _spr)

def rolling_dominant_cycle(s, w):
    """Trend Dim 3 — dominant cycle length (bars) from spectral peak."""
    def _dc(y):
        f, pxx = sp_signal.periodogram(y - y.mean())
        if len(f) < 2:
            return np.nan
        peak = pxx[1:].argmax() + 1
        return 1.0 / f[peak] if f[peak] > 0 else np.nan
    return _rolling_apply(s, w, _dc)

def regime_duration(s):
    """Trend Dim 4 — consecutive bars in same direction (run length)."""
    direction = np.sign(s.diff())
    count, runs = 0.0, []
    for d in direction:
        if np.isnan(d) or d == 0:
            count = 0.0
        elif len(runs) == 0 or np.sign(count) == d:
            count += d
        else:
            count = d
        runs.append(count)
    return pd.Series(runs, index=s.index)

def efficiency_ratio(s, w):
    direction  = (s - s.shift(w)).abs()
    volatility = s.diff().abs().rolling(w).sum()
    return direction / volatility.replace(0, np.nan)

def hurst_exp(s, w, lags=10):
    def _h(y):
        rs_vals = []
        for lag in range(2, lags + 1):
            sub  = y[:lag]
            dev  = np.cumsum(sub - sub.mean())
            rng  = dev.max() - dev.min()
            std  = sub.std()
            if std > 0:
                rs_vals.append(rng / std)
        if len(rs_vals) < 2:
            return np.nan
        h, *_ = _lr(np.log(range(2, len(rs_vals) + 2)), np.log(rs_vals))
        return h
    return _rolling_apply(s, w, _h)

def shannon_entropy(s, w, bins=10):
    rets = s.pct_change()
    def _ent(y):
        counts, _ = np.histogram(y[~np.isnan(y)], bins=bins)
        p = counts / (counts.sum() + 1e-12)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return _rolling_apply(rets, w, _ent)

def tema(s, w):
    e1 = s.ewm(span=w, adjust=False).mean()
    e2 = e1.ewm(span=w, adjust=False).mean()
    e3 = e2.ewm(span=w, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3

def kalman_price(s, q=1e-5, r=1e-3):
    x, p, out = s.iloc[0], 1.0, []
    for obs in s:
        p_pred = p + q
        k      = p_pred / (p_pred + r)
        x      = x + k * (obs - x)
        p      = (1 - k) * p_pred
        out.append(x)
    return pd.Series(out, index=s.index)

def hma(s, w):
    half = s.ewm(span=max(w // 2, 2), adjust=False).mean()
    full = s.ewm(span=w,              adjust=False).mean()
    return (2 * half - full).ewm(span=max(int(np.sqrt(w)), 2), adjust=False).mean()

# ── Compute all new features ──────────────────────────────────────────────────
try:
    pf = pd.read_csv(PRICE_CSV, parse_dates=[PRICE_DATE_COL])
    pf = pf.sort_values(PRICE_DATE_COL).reset_index(drop=True)
    cl, hi, lo = pf[PRICE_CLOSE_COL], pf[PRICE_HIGH_COL], pf[PRICE_LOW_COL]
    atr14 = (hi - lo).rolling(14).mean()   # simplified ATR

    print(f"✅ Price data loaded: {len(pf):,} bars  ({pf[PRICE_DATE_COL].iloc[0].date()} → "
          f"{pf[PRICE_DATE_COL].iloc[-1].date()})")

    # ── Trend Dimension 2: Quadratic curvature ────────────────────────────────
    pf["quad_curve_20"] = rolling_quad_curvature(cl, 20)
    pf["quad_curve_50"] = rolling_quad_curvature(cl, 50)

    # ── Trend Dimension 3: Spectral trend ─────────────────────────────────────
    pf["spectral_power_ratio_50"] = rolling_spectral_power_ratio(cl, 50)
    pf["dominant_cycle_50"]       = rolling_dominant_cycle(cl, 50)

    # ── Trend Dimension 4: Regime duration ────────────────────────────────────
    pf["regime_duration"] = regime_duration(cl)

    # ── Building blocks for ratios ────────────────────────────────────────────
    s10   = rolling_slope(cl, 10)
    s30   = rolling_slope(cl, 30)
    s50   = rolling_slope(cl, 50)
    r2_30 = rolling_r2(cl, 30)
    er20  = efficiency_ratio(cl, 20)
    h50   = hurst_exp(cl, 50)
    ent30 = shannon_entropy(cl, 30)
    sma30 = cl.rolling(30).mean()
    klm   = kalman_price(cl)
    tema20 = tema(cl, 20)

    don_hi   = cl.rolling(20).max()
    don_lo   = cl.rolling(20).min()
    don_bo   = pd.Series(
        np.where(cl >= don_hi, 1.0, np.where(cl <= don_lo, -1.0, 0.0)),
        index=cl.index)

    dm_up  = hi.diff().clip(lower=0)
    dm_dn  = (-lo.diff()).clip(lower=0)
    adx    = ((dm_up.where(dm_up > dm_dn, 0) - dm_dn.where(dm_dn > dm_up, 0)).abs()
              / (dm_up + dm_dn + 1e-9)).rolling(14).mean()

    # ── Top-10 ratios ─────────────────────────────────────────────────────────
    pf["ratio_01_accel"]        = s10   / s30.replace(0, np.nan)         # slope_10/slope_30
    pf["ratio_02_trend_snr"]    = s30   / atr14.replace(0, np.nan)       # slope_30/ATR_14
    pf["ratio_03_eff_slope"]    = s30   * er20                            # slope_30×ER_20
    pf["ratio_04_pers_slope"]   = s30   * h50                             # slope_30×hurst_50
    pf["ratio_05_struct_chaos"] = r2_30 / ent30.replace(0, np.nan)       # R²/entropy
    pf["ratio_06_kalman_sma"]   = klm   / sma30.replace(0, np.nan)       # kalman/SMA
    pf["ratio_07_tema_kalman"]  = tema20/ klm.replace(0, np.nan)         # TEMA/kalman
    pf["ratio_08_convexity"]    = s10   - s50                             # slope_10-slope_50
    pf["ratio_09_adx_entropy"]  = adx   / ent30.replace(0, np.nan)       # ADX/entropy
    pf["ratio_10_breakout_er"]  = don_bo/ er20.replace(0, np.nan)        # donchian_bo/ER

    NEW_FEATURES = [
        "quad_curve_20", "quad_curve_50",
        "spectral_power_ratio_50", "dominant_cycle_50",
        "regime_duration",
        "ratio_01_accel",       "ratio_02_trend_snr",
        "ratio_03_eff_slope",   "ratio_04_pers_slope",
        "ratio_05_struct_chaos","ratio_06_kalman_sma",
        "ratio_07_tema_kalman", "ratio_08_convexity",
        "ratio_09_adx_entropy", "ratio_10_breakout_er",
    ]
    LABELS = {
        "quad_curve_20"         : "Dim 2  — Quadratic curvature (w=20)",
        "quad_curve_50"         : "Dim 2  — Quadratic curvature (w=50)",
        "spectral_power_ratio_50": "Dim 3  — Spectral power ratio (Ehlers, w=50)",
        "dominant_cycle_50"     : "Dim 3  — Dominant cycle period bars (w=50)",
        "regime_duration"       : "Dim 4  — Regime run length (consecutive bars)",
        "ratio_01_accel"        : "Ratio 1  slope_10 / slope_30        [acceleration]",
        "ratio_02_trend_snr"    : "Ratio 2  slope_30 / ATR_14          [trend SNR]",
        "ratio_03_eff_slope"    : "Ratio 3  slope_30 × ER_20           [eff-weighted slope]",
        "ratio_04_pers_slope"   : "Ratio 4  slope_30 × hurst_50        [pers-weighted slope]",
        "ratio_05_struct_chaos" : "Ratio 5  R² / entropy               [structure vs chaos]",
        "ratio_06_kalman_sma"   : "Ratio 6  kalman / SMA_30            [dynamic vs static]",
        "ratio_07_tema_kalman"  : "Ratio 7  TEMA_20 / kalman           [curvature disagree]",
        "ratio_08_convexity"    : "Ratio 8  slope_10 − slope_50        [trend convexity]",
        "ratio_09_adx_entropy"  : "Ratio 9  ADX / entropy              [structured direction]",
        "ratio_10_breakout_er"  : "Ratio 10 donchian_breakout / ER     [clean breakout]",
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*76}")
    print("  NEW TREND DIMENSIONS & RATIOS — DESCRIPTIVE STATS")
    print(f"{'═'*76}")
    print(f"  {'FEATURE':<28}  {'MEAN':>9}  {'STD':>9}  {'MIN':>9}  {'MAX':>9}  {'NaN%':>5}")
    print(f"  {'─'*74}")
    for feat in NEW_FEATURES:
        col = pf[feat]
        nan_pct = col.isna().mean() * 100
        if col.notna().any():
            print(f"  {feat:<28}  {col.mean():>9.4f}  {col.std():>9.4f}  "
                  f"{col.min():>9.4f}  {col.max():>9.4f}  {nan_pct:>4.1f}%")
        else:
            print(f"  {feat:<28}  {'— all NaN —':>39}  {nan_pct:>4.1f}%")
    print(f"\n  Legend:")
    for feat, lbl in LABELS.items():
        print(f"    {feat:<28} → {lbl}")

    # ── Visualise new features ────────────────────────────────────────────────
    ratio_cols = [c for c in NEW_FEATURES if c.startswith("ratio_")]
    dim_cols   = [c for c in NEW_FEATURES if not c.startswith("ratio_")]

    fig_new, axes_new = plt.subplots(
        len(NEW_FEATURES), 1,
        figsize=(14, 2.5 * len(NEW_FEATURES)),
        sharex=True)
    if len(NEW_FEATURES) == 1:
        axes_new = [axes_new]

    x_idx = pf[PRICE_DATE_COL] if PRICE_DATE_COL in pf.columns else pf.index
    for ax, feat in zip(axes_new, NEW_FEATURES):
        col = pf[feat]
        color = "#e63946" if feat.startswith("ratio_") else "#457b9d"
        ax.plot(x_idx, col, lw=0.9, color=color, alpha=0.85)
        ax.axhline(0, ls=":", lw=0.7, color="gray")
        ax.set_ylabel(feat.replace("ratio_", "R").replace("_", " "),
                      fontsize=7, rotation=0, ha="right", labelpad=100)
        ax.tick_params(axis="both", labelsize=6)

    axes_new[0].set_title(
        "New Trend Dimensions (blue) & Ratios (red)",
        fontweight="bold", fontsize=11)
    fig_new.tight_layout()
    plt.show()

    # ── Export new features ───────────────────────────────────────────────────
    from datetime import datetime as _dt
    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    out_new = f"/content/drive/MyDrive/New_Trend_Features_{_ts}.csv"
    pf[[PRICE_DATE_COL] + NEW_FEATURES].to_csv(out_new, index=False)
    print(f"\n✅ New features saved → {out_new}")

except FileNotFoundError:
    print(f"⚠️  Price file not found: {PRICE_CSV}")
    print("   Update PRICE_CSV at the top of this cell and re-run.")


# %% [10] EXPORT FINAL ROSTERS TO CSV  (renumbered from 9)
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

out_rows = []
for brain, roster in final_rosters.items():
    roster_copy = roster.copy().reset_index()
    roster_copy.rename(columns={"index": "Rank"}, inplace=True)
    roster_copy["Brain"] = brain
    out_rows.append(roster_copy)

final_df = pd.concat(out_rows, ignore_index=True)

out_path = f"/content/drive/MyDrive/Sovereign_Final_Roster_{ts}.csv"
final_df.to_csv(out_path, index=False)
print(f"✅ Final roster saved → {out_path}")

# also display as HTML table
display(HTML(
    final_df[["Brain","Rank","Feature","Score","Avg_INorm",
              "Avg_Stab","Avg_UV","Persistence_pct","Is_Locked"]]
    .to_html(index=False, float_format="{:.4f}".format)
))


# %% [11] ITERATION SUFFICIENCY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  ITERATION SUFFICIENCY SUMMARY")
print("═"*70)

all_sufficient = True
for brain in BRAINS:
    v = verdicts[brain]
    status = v["verdict"]
    all_sufficient = all_sufficient and ("SUFFICIENT" in status)
    icon_rho  = "✅" if v["pass_rho"]     else "❌"
    icon_ovl  = "✅" if v["pass_overlap"] else "❌"
    icon_cv   = "✅" if v["pass_cv"]      else "❌"
    print(f"\n  {brain}")
    print(f"    Rank stability  (ρ ≥ 0.85)  : {icon_rho} ρ = {v['last_rho']:.4f}")
    print(f"    Slot stability  (≥16/19)    : {icon_ovl} overlap = {v['last_overlap']}/19")
    print(f"    Score variance  (CV ≤ 0.20) : {icon_cv} median CV = {v['median_cv']:.4f}")
    print(f"    ─ {status} ─", end="")
    if "MORE NEEDED" in status:
        print(f"  (recommend ≥ {v['recommend_n']} total iterations)")
    else:
        print()

print()
if all_sufficient:
    print("  ✅ ALL BRAINS CONVERGED — current iteration count is sufficient.")
else:
    worst = max(verdicts.values(), key=lambda v: v["recommend_n"])
    print(f"  ⚠️  NOT ALL BRAINS CONVERGED — run at least "
          f"{worst['recommend_n']} total iterations and re-assess.")
print("═"*70)
