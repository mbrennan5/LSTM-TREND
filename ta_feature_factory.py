import datetime
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# ### BLOCK 3: FEATURE FACTORY — uses Numba kernels throughout
# Top-level function required for ProcessPoolExecutor pickling.
# ==============================================================================

def generate_factory_features_v2(df):
    df = df.copy()
    df['hlc3']    = (df['high'] + df['low'] + df['close']) / 3
    df['T_FINAL'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    hlc = df['hlc3'].values.astype(np.float64)
    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    cl  = df['close'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    idx = df.index
    # ── Moving averages ────────────────────────────────────────────────────────
    ema10   = pd.Series(hlc, index=idx).ewm(span=10).mean().values
    ema10_2 = pd.Series(ema10, index=idx).ewm(span=10).mean().values
    ema10_3 = pd.Series(ema10_2, index=idx).ewm(span=10).mean().values
    tema_10 = 3*ema10 - 3*ema10_2 + ema10_3
    sma_5   = pd.Series(hlc, index=idx).rolling(5).mean().values
    sma_20  = pd.Series(hlc, index=idx).rolling(20).mean().values
    ema30   = pd.Series(hlc, index=idx).ewm(span=30).mean().values
    ema30_2 = pd.Series(ema30, index=idx).ewm(span=30).mean().values
    ema30_3 = pd.Series(ema30_2, index=idx).ewm(span=30).mean().values
    tema_30 = 3*ema30 - 3*ema30_2 + ema30_3
    # WMA via Numba — replaces two rolling().apply(lambda) calls
    wma1    = _rolling_wma(hlc, 10)
    wma2    = _rolling_wma(hlc, 21)
    hma_raw = 2 * wma1 - wma2
    hma_21  = pd.Series(hma_raw, index=idx).rolling(5).mean().values
    kalman = _kalman_numba(hlc)
    # ── Efficiency / trend strength ────────────────────────────────────────────
    hlc_s        = pd.Series(hlc, index=idx)
    er_20        = (hlc_s.diff(20).abs() /
                    (hlc_s.diff().abs().rolling(20).sum() + 1e-9)).values
    vidya_cmo_20 = (hlc_s.diff().rolling(20).sum() /
                    (hlc_s.diff().abs().rolling(20).sum() + 1e-9)).values
    # ── Numba rolling functions ────────────────────────────────────────────────
    r_sq_30    = _rolling_r_sq(hlc, 30)
    hurst_50   = _rolling_hurst(hlc, 50)
    shannon_20 = _rolling_shannon(hlc, 20)
    cog_20     = _rolling_cog(hlc, 20)
    # ── Multi-window linear slopes ─────────────────────────────────────────────
    linreg_10        = _rolling_linslope(hlc, 10)
    linreg_20        = _rolling_linslope(hlc, 20)
    linreg_30        = _rolling_linslope(hlc, 30)
    linreg_60        = _rolling_linslope(hlc, 60)
    slope_std        = np.nanstd(linreg_30) + 1e-9
    logistic_prob_30 = 1.0 / (1.0 + np.exp(-linreg_30 / slope_std))
    # ── MTSI (Modified True Strength Index via VWAP anchor) ───────────────────
    _tp_v  = pd.Series(hlc * vol, index=idx)
    _vol_s = pd.Series(vol, index=idx)
    _cl_s  = pd.Series(cl,  index=idx)
    mtsi   = (_cl_s - (_tp_v.rolling(2).sum() /
                       (_vol_s.rolling(2).sum() + 1e-9))).ewm(span=3).mean().values
    # ── ADX ────────────────────────────────────────────────────────────────────
    cl_prev  = np.roll(cl, 1); cl_prev[0] = cl[0]
    tr       = np.maximum(hi - lo,
               np.maximum(np.abs(hi - cl_prev), np.abs(lo - cl_prev)))
    atr_14   = pd.Series(tr, index=idx).rolling(14).mean().values
    hi_prev  = np.roll(hi, 1); hi_prev[0] = hi[0]
    lo_prev  = np.roll(lo, 1); lo_prev[0] = lo[0]
    plus_dm  = np.where((hi - hi_prev) > (lo_prev - lo),
                        np.maximum(hi - hi_prev, 0), 0).astype(np.float64)
    minus_dm = np.where((lo_prev - lo) > (hi - hi_prev),
                        np.maximum(lo_prev - lo, 0), 0).astype(np.float64)
    pdi14 = 100 * (pd.Series(plus_dm,  index=idx).rolling(14).mean() /
                   (pd.Series(atr_14,  index=idx) + 1e-9))
    mdi14 = 100 * (pd.Series(minus_dm, index=idx).rolling(14).mean() /
                   (pd.Series(atr_14,  index=idx) + 1e-9))
    adx_14 = (100 * np.abs(pdi14 - mdi14) /
              (pdi14 + mdi14 + 1e-9)).rolling(14).mean().values
    # ── Dispersion — vectorised ────────────────────────────────────────────────
    sma_30 = pd.Series(hlc, index=idx).rolling(30).mean().values
    d_sma  = hlc / (sma_30  + 1e-9) - 1
    d_tema = hlc / (tema_30 + 1e-9) - 1
    d_kal  = hlc / (kalman  + 1e-9) - 1
    dispersion_30 = np.std(np.stack([d_sma, d_tema, d_kal], axis=1), axis=1)
    # ── Donchian / Aroon ──────────────────────────────────────────────────────
    hi_s             = pd.Series(hi, index=idx)
    donchian_high_20 = (hi_s / hi_s.rolling(20).max() - 1).values
    donchian_high_50 = (hi_s / hi_s.rolling(50).max() - 1).values
    aroon_up_25      = hi_s.rolling(25).apply(
        lambda x: float(np.argmax(x)) / 25, raw=True
    ).values
    # ── Pre-transform Price MAs: hlc3 / MA - 1 ────────────────────────────────
    tema_10_pct = hlc / (tema_10 + 1e-9) - 1
    tema_30_pct = hlc / (tema_30 + 1e-9) - 1
    sma_5_pct   = hlc / (sma_5   + 1e-9) - 1
    sma_20_pct  = hlc / (sma_20  + 1e-9) - 1
    hma_21_pct  = hlc / (hma_21  + 1e-9) - 1
    kalman_pct  = hlc / (kalman  + 1e-9) - 1
    # ── VHF (Vertical Horizontal Filter — needed for vhf_adx_ratio) ───────────
    lo_s        = pd.Series(lo, index=idx)
    cl_diff_abs = pd.Series(np.abs(np.diff(cl, prepend=cl[0])), index=idx)
    vhf_28      = ((hi_s.rolling(28).max() - lo_s.rolling(28).min()) /
                   (cl_diff_abs.rolling(28).sum() + 1e-9)).values
    # ── Ratio features ────────────────────────────────────────────────────────
    kalman_sma_ratio   = kalman  / (sma_20  + 1e-9) - 1
    tema_kalman_ratio  = tema_30 / (kalman  + 1e-9) - 1
    exhaustion         = (hlc - kalman) / (atr_14   + 1e-9)
    # exhaustion_60: scale-invariant distance from 60-bar high (Tier 1)
    _hi60 = pd.Series(hlc, index=idx).rolling(60).max().values
    _lo60 = pd.Series(hlc, index=idx).rolling(60).min().values
    exhaustion_60      = (_hi60 - hlc) / (_hi60 - _lo60 + 1e-9)
    # ratio_acc: slope acceleration — fast slope / slow slope (slope_10 / slope_20)
    ratio_acc          = linreg_10 / (np.abs(linreg_20) + 1e-9) * np.sign(linreg_20)
    # ratio_snr: signal-to-noise — slope magnitude / ATR noise (slope_20 / ATR_14)
    ratio_snr          = np.abs(linreg_20) / (atr_14 + 1e-9)
    # curvature_diff: second-order geometry — slope_10 minus slope_60
    curvature_diff     = linreg_10 - linreg_60
    cycle_vs_trend     = np.abs(cog_20) / (r_sq_30 + 1e-9)
    ratio_eff_slope    = er_20 * np.sign(linreg_30)
    # ratio_pers_slope: memory-weighted velocity — Hurst-50 × slope_20
    ratio_pers_slope   = hurst_50 * np.sign(linreg_20)
    ratio_struct       = (tema_10_pct - sma_20_pct) / (np.abs(sma_20_pct) + 1e-9)
    adx_entropy_ratio  = adx_14  / (shannon_20 + 1e-9)
    # ratio_breakout_eff: breakout conviction — Donchian-20 / ER-20
    ratio_breakout_eff = np.abs(donchian_high_20) / (er_20 + 1e-9)
    r_sq_hurst_ratio   = r_sq_30  / (hurst_50   + 1e-9)
    vhf_adx_ratio      = vhf_28   / (adx_14     + 1e-9)
    # ── Z-lens group ──────────────────────────────────────────────────────────
    Z_LENS_INDICATORS = {
        'er_20':            pd.Series(er_20,            index=idx),
        'vidya_cmo_20':     pd.Series(vidya_cmo_20,     index=idx),
        'r_sq_30':          pd.Series(r_sq_30,          index=idx),
        'hurst_50':         pd.Series(hurst_50,         index=idx),
        'shannon_20':       pd.Series(shannon_20,       index=idx),
        'adx_14':           pd.Series(adx_14,           index=idx),
        'logistic_prob_30': pd.Series(logistic_prob_30, index=idx),
        'aroon_up_25':      pd.Series(aroon_up_25,      index=idx),
        'donchian_high_50': pd.Series(donchian_high_50, index=idx),
        'dispersion_30':    pd.Series(dispersion_30,    index=idx),
        'lr_slope_30':      pd.Series(linreg_30,        index=idx),
        'tema_10_pct':      pd.Series(tema_10_pct,      index=idx),
        'tema_30_pct':      pd.Series(tema_30_pct,      index=idx),
        'sma_5_pct':        pd.Series(sma_5_pct,        index=idx),
        'sma_20_pct':       pd.Series(sma_20_pct,       index=idx),
        'hma_21_pct':       pd.Series(hma_21_pct,       index=idx),
        'kalman_pct':          pd.Series(kalman_pct,          index=idx),
        'kalman_sma_ratio':    pd.Series(kalman_sma_ratio,    index=idx),
        'tema_kalman_ratio':   pd.Series(tema_kalman_ratio,   index=idx),
        'exhaustion':          pd.Series(exhaustion,          index=idx),
        'exhaustion_60':       pd.Series(exhaustion_60,       index=idx),
        'ratio_acc':           pd.Series(ratio_acc,           index=idx),
        'ratio_snr':           pd.Series(ratio_snr,           index=idx),
        'curvature_diff':      pd.Series(curvature_diff,      index=idx),
        'cycle_vs_trend':      pd.Series(cycle_vs_trend,      index=idx),
        'ratio_eff_slope':     pd.Series(ratio_eff_slope,     index=idx),
        'ratio_pers_slope':    pd.Series(ratio_pers_slope,    index=idx),
        'ratio_struct':        pd.Series(ratio_struct,        index=idx),
        'adx_entropy_ratio':   pd.Series(adx_entropy_ratio,   index=idx),
        'ratio_breakout_eff':  pd.Series(ratio_breakout_eff,  index=idx),
        'r_sq_hurst_ratio':    pd.Series(r_sq_hurst_ratio,    index=idx),
        'vhf_adx_ratio':       pd.Series(vhf_adx_ratio,       index=idx),
    }
    df_features = pd.DataFrame(Z_LENS_INDICATORS)
    df_features['T_FINAL'] = df['T_FINAL'].values
    return df_features


def run_lookback_tester():
    # --- PHASE 1: User Inputs ---
    print("--- TA Feature Factory: Lookback Optimization Tester ---")

    # Best Practice: Test multiple candidate periods (e.g., Source 5's 72-day benchmark)
    lookbacks_input = input("Enter a CSV list of lookback periods to test (e.g. 10, 30, 72): ")
    lookbacks = [int(x.strip()) for x in lookbacks_input.split(',')]

    # Best Practice: Use a diverse symbol set to ensure generalizability
    num_symbols = int(input("Enter the number of symbols to test: "))

    # Best Practice: Iterative optimization (GA/BO) to explore the search space
    iterations = int(input("Enter number of iterations to run: "))

    # --- PHASE 2: Random Period Selection ---
    # Requirement: Choose a random 3-year period within the last 15 years
    current_year = 2026
    start_search_range = current_year - 15
    end_search_range   = current_year - 3  # To allow for a 3-year window

    random_start_year  = random.randint(start_search_range, end_search_range)
    random_start_month = random.randint(1, 12)
    start_date = datetime.date(random_start_year, random_start_month, 1)
    end_date   = start_date + datetime.timedelta(days=3 * 365)

    print(f"\n--- Phase 3: Selection Strategy ---")
    print(f"Target Window: {start_date} to {end_date} (3 Years)")
    print(f"Testing {len(lookbacks)} lookback configurations across {num_symbols} symbols.")

    # --- PHASE 4: Core Optimization Engine ---
    # Based on the GA-LSTM-BO and LASSO frameworks
    for iteration in range(iterations):
        for n in lookbacks:
            # 1. Obtain raw_data for the selected period and symbols here
            # 2. Feature Preprocessing: call generate_factory_features_v2(raw_data)
            #    df_features = generate_factory_features_v2(raw_data)
            # 3. Outlier Removal via Z-Score (rolling window lens)
            # 4. LASSO Compression: retain only non-zero critical variables
            #    This determines if lookback 'n' produces "Critical Predictor Variables"
            # 5. Evaluate: Calculate Sharpe Ratio or MSE for 1-day forward target
            pass

    # Results identify which period consistently maximizes the portfolio Sharpe Ratio
    print("\n[Diagnostic Ready] Tester will now evaluate feature significance for the 1-day forward target.")


if __name__ == "__main__":
    run_lookback_tester()
