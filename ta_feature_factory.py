# ==============================================================================
# ta_feature_factory.py  — Joint Lookback × Z-Lens Optimization
# Sovereign Titan v3.20 "Ron Harper 4.30" — Numba Block 2 + parameterized Block 3
#
# Mandate (Sovereign framework):
#   Never optimize the seed indicator in isolation.
#   indicator_n and z_n are COUPLED hyperparameters (Joint Optimization).
#   Score = 0.70 × Impact  +  0.15 × Uniqueness  +  0.15 × Stability
#
# At startup the tester prints a numbered feature catalog and asks which
# features to include in this run — keeping each test focused and fast.
# ==============================================================================
import os
import gc
import datetime
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler
from numba import jit
from tqdm.auto import tqdm

N_FEATURE_WORKERS = max(1, (os.cpu_count() or 2))
print(f"[SYSTEM] Feature workers: {N_FEATURE_WORKERS}")

TITAN_SYMBOLS = [
    'AA','AAL','AAPL','ABNB','ACWI','AEM','AFRM','AI','ALAB','ALB','AMAT','AMD','AMZN',
    'ANET','APA','APH','ARKK','AVGO','BA','BABA','BAC','BKR','BLDR','C','CARR','CAT',
    'CCJ','CCL','CE','CELH','CLF','CLSK','CMG','CNC','CPRT','CRM','CSCO','CSX','CVS',
    'CVX','DAL','DDOG','DHR','DIA','DIS','DKNG','DLTR','DOW','DVN','DXCM','EA','EBAY',
    'EEM','EMR','EQT','EWJ','EWT','EWW','EWY','EWZ','EXC','F','FANG','FCX','FITB',
    'FTNT','FTV','FXI','GBTC','GDX','GDXJ','GEHC','GFS','GIS','GOOG','GOOGL','GS',
    'HAL','HOOD','HPE','HPQ','HWM','IAU','IBM','IGV','IJH','IJR','INTC','IP','IR',
    'IWM','IYR','JNJ','KDP','KMI','KO','KRE','KWEB','LOW','LRCX','LUV','LVS','LYFT',
    'MAR','MARA','MCHP','MGM','MNST','MPC','MRK','MRNA','MRVL','MS','MSFT','MSTR',
    'MU','NCLH','NEE','NEM','NKE','NUE','NVDA','NVO','NXPI','ON','ORCL','OXY','PANW',
    'PCAR','PDD','PEP','PFE','PINS','PLTR','PYPL','QCOM','QQQ','QQQM','RBLX','RIOT',
    'RIVN','RTX','SBUX','SCHW','SHOP','SJM','SLB','SLV','SMCI','SMH','SNAP','SNOW',
    'SOFI','SOXX','SPLG','SPY','TER','TGT','TJX','TLT','TMUS','TQQQ','TSCO','TSLA',
    'TTD','TTWO','TWLO','TXN','U','UAL','UBER','UPS','USB','USO','VLO','VNQ','VRT',
    'VST','VT','VTR','WMT','WYNN','XBI','XLB','XLC','XLE','XLF','XLI','XLK','XLP',
    'XLRE','XLU','XLV','XLY','XOM','XOP','XRT',
]

# ==============================================================================
# FEATURE CATALOG  — numbered reference for the interactive selector
# (num, key, description, family)
# ==============================================================================
FEATURE_CATALOG = [
    ( 1, 'er',               'Efficiency Ratio (ER)',                     'Trend'),
    ( 2, 'vidya_cmo',        'VIDYA CMO (Directional Efficiency)',         'Trend'),
    ( 3, 'r_sq',             'R-squared (Trend Linearity)',                'Trend'),
    ( 4, 'hurst',            'Hurst Exponent (Long Memory)',               'Regime'),
    ( 5, 'shannon',          'Shannon Entropy (Market Noise)',             'Regime'),
    ( 6, 'adx',              'ADX (Trend Strength)',                       'Trend'),
    ( 7, 'logistic_prob',    'Logistic Prob (Sigmoid of LR Slope)',        'Momentum'),
    ( 8, 'aroon_up',         'Aroon Up (Recency of Rolling High)',         'Momentum'),
    ( 9, 'donchian_l',       'Donchian High Long (Breakout Position)',     'Breakout'),
    (10, 'dispersion',       'Dispersion (3-MA Spread)',                   'Volatility'),
    (11, 'lr_slope',         'Linear Regression Slope',                    'Momentum'),
    (12, 'tema_s_pct',       'TEMA Short  %  (hlc / TEMA_s − 1)',         'Price-MA'),
    (13, 'tema_m_pct',       'TEMA Primary % (hlc / TEMA_m − 1)',         'Price-MA'),
    (14, 'sma_xs_pct',       'SMA XShort %  (hlc / SMA_xs − 1)',         'Price-MA'),
    (15, 'sma_m_pct',        'SMA Primary % (hlc / SMA_m − 1)',           'Price-MA'),
    (16, 'hma_m_pct',        'HMA Primary % (hlc / HMA_m − 1)',           'Price-MA'),
    (17, 'kalman_pct',       'Kalman %       (hlc / Kalman − 1)',         'Price-MA'),
    (18, 'kalman_sma_ratio', 'Kalman / SMA Ratio',                        'Ratio'),
    (19, 'tema_kal_ratio',   'TEMA / Kalman Ratio',                       'Ratio'),
    (20, 'exhaustion',       'Exhaustion  (hlc − Kalman) / ATR',          'Mean-Rev'),
    (21, 'exhaustion_xl',    'Exhaustion XL (60-bar range position)',      'Mean-Rev'),
    (22, 'ratio_acc',        'Slope Acceleration  (slope_s / slope_m)',   'Ratio'),
    (23, 'ratio_snr',        'Signal-to-Noise     (|slope_m| / ATR)',     'Ratio'),
    (24, 'curvature_diff',   'Curvature Diff      (slope_s − slope_xl)',  'Ratio'),
    (25, 'cycle_vs_trend',   'Cycle vs Trend      (|CoG| / R²)',          'Ratio'),
    (26, 'ratio_eff_slope',  'Efficiency × Slope Direction',              'Ratio'),
    (27, 'ratio_pers_slope', 'Persistence × Slope (Hurst × sign)',        'Ratio'),
    (28, 'ratio_struct',     'Structure Ratio     (TEMA_s − SMA_m)',      'Ratio'),
    (29, 'adx_entropy',      'ADX / Entropy Ratio',                       'Ratio'),
    (30, 'breakout_eff',     'Breakout Efficiency (Donchian / ER)',       'Ratio'),
    (31, 'r_sq_hurst',       'R² / Hurst Ratio',                         'Ratio'),
    (32, 'vhf_adx',          'VHF / ADX Ratio',                          'Ratio'),
    (33, 'cog',              'Center of Gravity (Rolling Deviation)',      'Oscillator'),
]

# Build lookup dicts
_CATALOG_BY_NUM = {num: (key, desc, fam) for num, key, desc, fam in FEATURE_CATALOG}
_CATALOG_BY_KEY = {key: (num, desc, fam) for num, key, desc, fam in FEATURE_CATALOG}
ALL_FEATURE_KEYS = [key for _, key, _, _ in FEATURE_CATALOG]


def print_feature_catalog():
    """Print numbered feature catalog grouped by family."""
    families = {}
    for num, key, desc, fam in FEATURE_CATALOG:
        families.setdefault(fam, []).append((num, key, desc))

    print("\n" + "═" * 68)
    print("  FEATURE CATALOG  — enter numbers to include in this run")
    print("═" * 68)
    for fam, items in families.items():
        print(f"\n  [{fam}]")
        for num, key, desc in items:
            print(f"    {num:>2}.  {desc}")
    print()
    print("  Ranges OK:  e.g.  1,3,5-10,15,20-25")
    print("  Enter 'all' to include every feature")
    print("═" * 68)


def parse_feature_selection(raw: str) -> list:
    """Parse '1,3,5-10,all' → list of feature keys in catalog order."""
    raw = raw.strip().lower()
    if raw == 'all':
        return list(ALL_FEATURE_KEYS)
    selected_nums = set()
    for token in raw.split(','):
        token = token.strip()
        if '-' in token:
            lo, hi = token.split('-')
            selected_nums.update(range(int(lo), int(hi) + 1))
        elif token.isdigit():
            selected_nums.add(int(token))
    valid = sorted(n for n in selected_nums if n in _CATALOG_BY_NUM)
    return [_CATALOG_BY_NUM[n][0] for n in valid]


# ==============================================================================
# BLOCK 2: NUMBA JIT ROLLING KERNELS  (Sovereign Titan v3.20 — Ron Harper 4.30)
# cache=True saves compiled artifacts to disk — subsequent runs skip recompile.
# ==============================================================================
@jit(nopython=True, cache=True)
def _lin_slope_nb(y):
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = i - x_mean
        num += dx * (y[i] - y_mean)
        den += dx * dx
    return num / den if den != 0.0 else 0.0

@jit(nopython=True, cache=True)
def _rolling_linslope(arr, window):
    n = len(arr)
    out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _lin_slope_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _hurst_nb(y):
    n = len(y)
    if n < 2:
        return 0.5
    mean = 0.0
    for i in range(n):
        mean += y[i]
    mean /= n
    var = 0.0
    for i in range(n):
        var += (y[i] - mean) ** 2
    std = (var / n) ** 0.5
    if std < 1e-12:
        return 0.5
    r = np.log(std + 1e-9) / np.log(n)
    return r if not np.isnan(r) else 0.5

@jit(nopython=True, cache=True)
def _rolling_hurst(arr, window):
    n = len(arr)
    out = np.full(n, 0.5)
    for i in range(window - 1, n):
        out[i] = _hurst_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _cog_nb(y):
    n = len(y)
    if n < 2:
        return 0.0
    num = 0.0
    den = 0.0
    for i in range(n):
        w = float(i + 1)
        num += w * y[i]
        den += y[i]
    return -num / (den + 1e-9)

@jit(nopython=True, cache=True)
def _rolling_cog(arr, window):
    n = len(arr)
    out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _cog_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _shannon_nb(y, bins=10):
    n = len(y)
    if n < 2:
        return 0.0
    mn = y[0]
    mx = y[0]
    for i in range(1, n):
        if y[i] < mn:
            mn = y[i]
        if y[i] > mx:
            mx = y[i]
    if mx == mn:
        return 0.0
    counts = np.zeros(bins)
    for i in range(n):
        idx = int((y[i] - mn) / (mx - mn) * bins)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1.0
    entropy = 0.0
    for i in range(bins):
        p = counts[i] / n + 1e-9
        entropy -= p * np.log(p)
    return entropy

@jit(nopython=True, cache=True)
def _rolling_shannon(arr, window):
    n = len(arr)
    out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _shannon_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _r_sq_nb(y):
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = i - x_mean
        dy = y[i] - y_mean
        num   += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    r = num / ((den_x ** 0.5) * (den_y ** 0.5))
    return r * r

@jit(nopython=True, cache=True)
def _rolling_r_sq(arr, window):
    n = len(arr)
    out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _r_sq_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _rolling_wma(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    w_sum = window * (window + 1) / 2.0
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += arr[i - window + 1 + j] * (j + 1)
        out[i] = s / w_sum
    return out

@jit(nopython=True, cache=True)
def _kalman_numba(price, r=0.0001, q=0.001):
    x_hat = np.zeros_like(price)
    p     = np.zeros_like(price)
    x_hat[0] = price[0]
    p[0]     = 1.0
    for t in range(1, len(price)):
        p_minus   = p[t - 1] + q
        k         = p_minus / (p_minus + r)
        x_hat[t]  = x_hat[t - 1] + k * (price[t] - x_hat[t - 1])
        p[t]      = (1 - k) * p_minus
    return x_hat


def _warm_up_numba():
    dummy = np.random.randn(60).astype(np.float64)
    _rolling_linslope(dummy, 10)
    _rolling_hurst(dummy, 30)
    _rolling_cog(dummy, 20)
    _rolling_shannon(dummy, 20)
    _rolling_r_sq(dummy, 30)
    _rolling_wma(dummy, 10)
    _kalman_numba(dummy)
    print("✅ Numba kernels compiled and ready")

_warm_up_numba()


# ==============================================================================
# BLOCK 3: PARAMETERIZED FEATURE FACTORY
#
# indicator_n  — primary lookback; all indicator windows scale from this:
#     n_s  = max(5,  indicator_n // 2)   short
#     n_m  = max(10, indicator_n)         primary
#     n_l  = max(20, indicator_n * 2)     long
#     n_xl = max(30, indicator_n * 3)     extra-long
#     n_atr= max(7,  indicator_n // 2)    ATR/ADX
#
# z_n          — rolling window for the Kinematic lens (Z, Z-slope, SOS).
#                This is the ONLY lens window — it replaces the hardcoded
#                [10,30,60,90] grid so the joint search is clean.
#
# selected_features — list of keys from FEATURE_CATALOG to include.
#                     All intermediates are always computed; only selected
#                     seeds are passed through the Z-lens and returned.
# ==============================================================================
def generate_factory_features_v2(df, indicator_n=20, z_n=20,
                                  selected_features=None):
    if selected_features is None:
        selected_features = ALL_FEATURE_KEYS

    df = df.copy()
    df['hlc3']    = (df['high'] + df['low'] + df['close']) / 3
    df['T_FINAL'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    hlc = df['hlc3'].values.astype(np.float64)
    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    cl  = df['close'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    idx = df.index

    # ── Window family derived from indicator_n ─────────────────────────────────
    n_s   = max(5,  indicator_n // 2)
    n_m   = max(10, indicator_n)
    n_l   = max(20, indicator_n * 2)
    n_xl  = max(30, indicator_n * 3)
    n_atr = max(7,  indicator_n // 2)

    # ── Moving averages ────────────────────────────────────────────────────────
    ema_s    = pd.Series(hlc, index=idx).ewm(span=n_s).mean().values
    ema_s2   = pd.Series(ema_s,  index=idx).ewm(span=n_s).mean().values
    ema_s3   = pd.Series(ema_s2, index=idx).ewm(span=n_s).mean().values
    tema_s   = 3*ema_s - 3*ema_s2 + ema_s3

    sma_xs   = pd.Series(hlc, index=idx).rolling(max(3, n_s // 2)).mean().values
    sma_m    = pd.Series(hlc, index=idx).rolling(n_m).mean().values

    ema_m    = pd.Series(hlc, index=idx).ewm(span=n_m).mean().values
    ema_m2   = pd.Series(ema_m,  index=idx).ewm(span=n_m).mean().values
    ema_m3   = pd.Series(ema_m2, index=idx).ewm(span=n_m).mean().values
    tema_m   = 3*ema_m - 3*ema_m2 + ema_m3

    wma1     = _rolling_wma(hlc, n_s)
    wma2     = _rolling_wma(hlc, n_m)
    hma_raw  = 2 * wma1 - wma2
    hma_m    = pd.Series(hma_raw, index=idx).rolling(max(3, int(n_s ** 0.5))).mean().values

    kalman   = _kalman_numba(hlc)

    # ── Efficiency / trend strength ────────────────────────────────────────────
    hlc_s        = pd.Series(hlc, index=idx)
    er_m         = (hlc_s.diff(n_m).abs() /
                    (hlc_s.diff().abs().rolling(n_m).sum() + 1e-9)).values
    vidya_cmo_m  = (hlc_s.diff().rolling(n_m).sum() /
                    (hlc_s.diff().abs().rolling(n_m).sum() + 1e-9)).values

    # ── Numba rolling ──────────────────────────────────────────────────────────
    r_sq_m     = _rolling_r_sq(hlc, n_m)
    hurst_l    = _rolling_hurst(hlc, n_l)
    shannon_m  = _rolling_shannon(hlc, n_m)
    cog_m      = _rolling_cog(hlc, n_m)

    # ── Linear slopes ──────────────────────────────────────────────────────────
    linreg_s   = _rolling_linslope(hlc, n_s)
    linreg_m   = _rolling_linslope(hlc, n_m)
    linreg_xl  = _rolling_linslope(hlc, n_xl)
    slope_std  = np.nanstd(linreg_m) + 1e-9
    logistic_m = 1.0 / (1.0 + np.exp(-linreg_m / slope_std))

    # ── MTSI ──────────────────────────────────────────────────────────────────
    _tp_v  = pd.Series(hlc * vol, index=idx)
    _vol_s = pd.Series(vol, index=idx)
    _cl_s  = pd.Series(cl,  index=idx)
    _mtsi  = (_cl_s - (_tp_v.rolling(2).sum() /
                        (_vol_s.rolling(2).sum() + 1e-9))).ewm(span=3).mean().values  # noqa: F841

    # ── ADX ────────────────────────────────────────────────────────────────────
    cl_prev  = np.roll(cl, 1); cl_prev[0] = cl[0]
    tr       = np.maximum(hi - lo,
               np.maximum(np.abs(hi - cl_prev), np.abs(lo - cl_prev)))
    atr_n    = pd.Series(tr, index=idx).rolling(n_atr).mean().values
    hi_prev  = np.roll(hi, 1); hi_prev[0] = hi[0]
    lo_prev  = np.roll(lo, 1); lo_prev[0] = lo[0]
    plus_dm  = np.where((hi - hi_prev) > (lo_prev - lo),
                        np.maximum(hi - hi_prev, 0), 0).astype(np.float64)
    minus_dm = np.where((lo_prev - lo) > (hi - hi_prev),
                        np.maximum(lo_prev - lo, 0), 0).astype(np.float64)
    pdi_n = 100 * (pd.Series(plus_dm,  index=idx).rolling(n_atr).mean() /
                   (pd.Series(atr_n,   index=idx) + 1e-9))
    mdi_n = 100 * (pd.Series(minus_dm, index=idx).rolling(n_atr).mean() /
                   (pd.Series(atr_n,   index=idx) + 1e-9))
    adx_n = (100 * np.abs(pdi_n - mdi_n) /
             (pdi_n + mdi_n + 1e-9)).rolling(n_atr).mean().values

    # ── Dispersion ─────────────────────────────────────────────────────────────
    sma_l  = pd.Series(hlc, index=idx).rolling(n_l).mean().values
    d_sma  = hlc / (sma_l   + 1e-9) - 1
    d_tema = hlc / (tema_m  + 1e-9) - 1
    d_kal  = hlc / (kalman  + 1e-9) - 1
    dispersion_l = np.std(np.stack([d_sma, d_tema, d_kal], axis=1), axis=1)

    # ── Donchian / Aroon ──────────────────────────────────────────────────────
    hi_s         = pd.Series(hi, index=idx)
    lo_s         = pd.Series(lo, index=idx)
    donchian_m   = (hi_s / hi_s.rolling(n_m).max() - 1).values
    donchian_l   = (hi_s / hi_s.rolling(n_l).max() - 1).values
    aroon_up_m   = hi_s.rolling(n_m).apply(
        lambda x: float(np.argmax(x)) / len(x), raw=True
    ).values

    # ── Price MA ratios ────────────────────────────────────────────────────────
    tema_s_pct  = hlc / (tema_s  + 1e-9) - 1
    tema_m_pct  = hlc / (tema_m  + 1e-9) - 1
    sma_xs_pct  = hlc / (sma_xs  + 1e-9) - 1
    sma_m_pct   = hlc / (sma_m   + 1e-9) - 1
    hma_m_pct   = hlc / (hma_m   + 1e-9) - 1
    kalman_pct  = hlc / (kalman  + 1e-9) - 1

    # ── VHF ────────────────────────────────────────────────────────────────────
    cl_diff_abs = pd.Series(np.abs(np.diff(cl, prepend=cl[0])), index=idx)
    vhf_m       = ((hi_s.rolling(n_m).max() - lo_s.rolling(n_m).min()) /
                   (cl_diff_abs.rolling(n_m).sum() + 1e-9)).values

    # ── Ratio features ─────────────────────────────────────────────────────────
    kalman_sma_ratio   = kalman / (sma_m  + 1e-9) - 1
    tema_kal_ratio     = tema_m / (kalman + 1e-9) - 1
    exhaustion         = (hlc - kalman) / (atr_n  + 1e-9)
    _hi_xl             = pd.Series(hlc, index=idx).rolling(n_xl).max().values
    _lo_xl             = pd.Series(hlc, index=idx).rolling(n_xl).min().values
    exhaustion_xl      = (_hi_xl - hlc) / (_hi_xl - _lo_xl + 1e-9)
    ratio_acc          = linreg_s / (np.abs(linreg_m)  + 1e-9) * np.sign(linreg_m)
    ratio_snr          = np.abs(linreg_m) / (atr_n + 1e-9)
    curvature_diff     = linreg_s - linreg_xl
    cycle_vs_trend     = np.abs(cog_m) / (r_sq_m + 1e-9)
    ratio_eff_slope    = er_m * np.sign(linreg_m)
    ratio_pers_slope   = hurst_l * np.sign(linreg_m)
    ratio_struct       = (tema_s_pct - sma_m_pct) / (np.abs(sma_m_pct) + 1e-9)
    adx_entropy        = adx_n  / (shannon_m + 1e-9)
    breakout_eff       = np.abs(donchian_m) / (er_m  + 1e-9)
    r_sq_hurst         = r_sq_m  / (hurst_l  + 1e-9)
    vhf_adx            = vhf_m   / (adx_n    + 1e-9)

    # ── Full seed family (all 32 numeric indicators) ───────────────────────────
    ALL_SEEDS = {
        'er':               pd.Series(er_m,            index=idx),
        'vidya_cmo':        pd.Series(vidya_cmo_m,     index=idx),
        'r_sq':             pd.Series(r_sq_m,          index=idx),
        'hurst':            pd.Series(hurst_l,         index=idx),
        'shannon':          pd.Series(shannon_m,       index=idx),
        'adx':              pd.Series(adx_n,           index=idx),
        'logistic_prob':    pd.Series(logistic_m,      index=idx),
        'aroon_up':         pd.Series(aroon_up_m,      index=idx),
        'donchian_l':       pd.Series(donchian_l,      index=idx),
        'dispersion':       pd.Series(dispersion_l,    index=idx),
        'lr_slope':         pd.Series(linreg_m,        index=idx),
        'tema_s_pct':       pd.Series(tema_s_pct,      index=idx),
        'tema_m_pct':       pd.Series(tema_m_pct,      index=idx),
        'sma_xs_pct':       pd.Series(sma_xs_pct,      index=idx),
        'sma_m_pct':        pd.Series(sma_m_pct,       index=idx),
        'hma_m_pct':        pd.Series(hma_m_pct,       index=idx),
        'kalman_pct':       pd.Series(kalman_pct,      index=idx),
        'kalman_sma_ratio': pd.Series(kalman_sma_ratio, index=idx),
        'tema_kal_ratio':   pd.Series(tema_kal_ratio,  index=idx),
        'exhaustion':       pd.Series(exhaustion,      index=idx),
        'exhaustion_xl':    pd.Series(exhaustion_xl,   index=idx),
        'ratio_acc':        pd.Series(ratio_acc,       index=idx),
        'ratio_snr':        pd.Series(ratio_snr,       index=idx),
        'curvature_diff':   pd.Series(curvature_diff,  index=idx),
        'cycle_vs_trend':   pd.Series(cycle_vs_trend,  index=idx),
        'ratio_eff_slope':  pd.Series(ratio_eff_slope, index=idx),
        'ratio_pers_slope': pd.Series(ratio_pers_slope, index=idx),
        'ratio_struct':     pd.Series(ratio_struct,    index=idx),
        'adx_entropy':      pd.Series(adx_entropy,     index=idx),
        'breakout_eff':     pd.Series(breakout_eff,    index=idx),
        'r_sq_hurst':       pd.Series(r_sq_hurst,      index=idx),
        'vhf_adx':          pd.Series(vhf_adx,         index=idx),
    }

    # ── Apply Z-lens ONLY to selected features ────────────────────────────────
    # Lens produces the Physics Trio:
    #   z        → Level 3: Position  (where the indicator is relative to its history)
    #   z_s      → Level 3: Velocity  (how fast it's moving)
    #   z_sos    → Level 4: Acceleration / SOS  (is the velocity changing direction?)
    sel_set = set(selected_features)
    for name, series in ALL_SEEDS.items():
        if name not in sel_set:
            continue
        arr  = series.values.astype(np.float64)
        rm   = pd.Series(arr, index=idx).rolling(z_n).mean().values
        rs   = pd.Series(arr, index=idx).rolling(z_n).std().values
        z    = (arr - rm) / (rs + 1e-9)
        zs   = _rolling_linslope(z, z_n)
        zsos = _rolling_linslope(zs, z_n)
        df[f'z_{name}']     = z
        df[f'z_s_{name}']   = zs
        df[f'z_sos_{name}'] = zsos

    # ── CoG rolling pct group (only if 'cog' selected) ───────────────────────
    if 'cog' in sel_set:
        cog_arr = cog_m.astype(np.float64)
        for win in [n_s, n_m, n_l]:
            rm = pd.Series(cog_arr, index=idx).rolling(win).mean().values
            df[f'cog_pct_{win}'] = cog_arr / (rm + 1e-9) - 1

    return (df.replace([np.inf, -np.inf], np.nan)
              .ffill()
              .dropna(subset=['T_FINAL'])
              .fillna(0))


# ==============================================================================
# BLOCK 4: PARALLEL DATA LOADER  (local-compatible, date-range aware)
# ==============================================================================
def _process_symbol_worker(args):
    """Top-level for ProcessPoolExecutor (must be picklable)."""
    symbol, raw_dict, indicator_n, z_n, selected_features = args
    try:
        raw_df = pd.DataFrame(raw_dict)
        raw_df.index = pd.to_datetime(raw_df.index)
        min_bars = max(120, indicator_n * 4)
        if len(raw_df) < min_bars:
            return None
        processed = generate_factory_features_v2(
            raw_df, indicator_n, z_n, selected_features
        )
        if processed.empty:
            return None
        processed['symbol'] = symbol
        return processed.reset_index()
    except Exception:
        return None


def fetch_data(symbol, start=None, end=None):
    try:
        kwargs = dict(progress=False)
        if start and end:
            kwargs['start'] = str(start)
            kwargs['end']   = str(end)
        else:
            kwargs['period'] = '3y'
        data = yf.download(symbol, interval='1d', **kwargs)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).lower() for c in data.columns]
        return data if 'close' in data.columns else None
    except Exception:
        return None


def load_hybrid_data_parallel(symbol_list, start_date=None, end_date=None,
                               indicator_n=20, z_n=20,
                               selected_features=None, dl_workers=20):
    """Download OHLCV + build features for a given (indicator_n, z_n, features) triple."""
    if selected_features is None:
        selected_features = ALL_FEATURE_KEYS

    raw_results = {}
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        fut_map = {
            pool.submit(fetch_data, sym, start_date, end_date): sym
            for sym in symbol_list
        }
        for fut in as_completed(fut_map):
            sym  = fut_map[fut]
            data = fut.result()
            min_bars = max(120, indicator_n * 4)
            if data is not None and len(data) >= min_bars:
                raw_results[sym] = data

    if not raw_results:
        return pd.DataFrame()

    work_items = [
        (sym, df.to_dict(), indicator_n, z_n, selected_features)
        for sym, df in raw_results.items()
    ]
    all_data = []
    try:
        with ProcessPoolExecutor(max_workers=N_FEATURE_WORKERS) as pool:
            futures = {
                pool.submit(_process_symbol_worker, item): item[0]
                for item in work_items
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    result = result.set_index(result.columns[0])
                    all_data.append(result)
    except Exception as e:
        print(f"  ⚠️  ProcessPool failed ({e}) — falling back to sequential")
        for item in work_items:
            result = _process_symbol_worker(item)
            if result is not None:
                result = result.set_index(result.columns[0])
                all_data.append(result)

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, axis=0)


# ==============================================================================
# JOINT SCORER: LASSO-based Sovereign Score for each (indicator_n, z_n) pair
# Score = 0.70 × Impact  +  0.15 × Uniqueness  +  0.15 × Stability
#   Impact     = val-set R²  of LassoCV predicting 1-day T_FINAL
#   Uniqueness = 1 − (nonzero_features / total_features)  [selectivity]
#   Stability  = normalised directional Sharpe on val set
# ==============================================================================
def _lasso_score(master_df):
    feat_cols = [
        c for c in master_df.columns
        if c.startswith('z_') or c.startswith('cog_pct_')
    ]
    if not feat_cols or 'T_FINAL' not in master_df.columns:
        return {'n_nonzero': 0, 'r2': 0.0, 'sharpe': 0.0, 'score': 0.0}

    X = (master_df[feat_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0)
         .values.astype(np.float32))
    y = master_df['T_FINAL'].values.astype(np.float32)

    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(X) < 100:
        return {'n_nonzero': 0, 'r2': 0.0, 'sharpe': 0.0, 'score': 0.0}

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    split    = int(len(X_scaled) * 0.8)
    X_tr, X_val = X_scaled[:split], X_scaled[split:]
    y_tr, y_val = y[:split], y[split:]

    lasso = LassoCV(cv=3, max_iter=2000, n_jobs=-1)
    lasso.fit(X_tr, y_tr)

    n_nonzero  = int(np.sum(lasso.coef_ != 0))
    r2         = float(max(0.0, lasso.score(X_val, y_val)))
    preds      = lasso.predict(X_val)
    signal     = np.sign(preds - 0.5)
    returns    = (y_val * 2 - 1) * signal
    sharpe     = float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))

    uniqueness = 1.0 - (n_nonzero / (len(feat_cols) + 1e-9))
    stability  = float(np.clip(sharpe / 5.0, 0.0, 1.0))

    score = 0.70 * r2 + 0.15 * uniqueness + 0.15 * stability
    return {
        'n_nonzero': n_nonzero,
        'r2':        r2,
        'sharpe':    sharpe,
        'score':     float(score),
    }


# ==============================================================================
# LOOKBACK TESTER: Joint Optimization over (indicator_n × z_n)
# ==============================================================================
def run_lookback_tester():
    print("\n" + "═" * 68)
    print("  TA Feature Factory — Joint Lookback × Z-Lens Optimizer")
    print("  Mandate: indicator_n and z_n optimized SIMULTANEOUSLY")
    print("═" * 68)

    # ── Step 0: Feature selection ──────────────────────────────────────────────
    print_feature_catalog()
    feat_input = input("Select features to optimize this run: ").strip()
    selected   = parse_feature_selection(feat_input)

    if not selected:
        print("⚠️  No valid features selected. Exiting.")
        return pd.DataFrame()

    print(f"\n  Selected {len(selected)} feature(s):")
    for key in selected:
        num, desc, fam = _CATALOG_BY_KEY[key]
        print(f"    {num:>2}. [{fam:<10}] {desc}")

    # ── Step 1: Search grid ────────────────────────────────────────────────────
    print()
    ind_input = input("Indicator lookback periods, CSV (e.g. 10,20,30): ").strip()
    indicator_lookbacks = [int(x) for x in ind_input.split(',') if x.strip().isdigit()]

    z_input = input("Z-lens window periods,       CSV (e.g. 10,20,30): ").strip()
    z_lens_periods = [int(x) for x in z_input.split(',') if x.strip().isdigit()]

    num_symbols = int(input("Symbols per iteration (e.g. 30):              ") or "30")
    iterations  = int(input("Iterations to run     (e.g. 5):               ") or "5")

    # ── Step 2: Random 3-year window within the last 15 years ─────────────────
    current_year       = 2026
    random_start_year  = random.randint(current_year - 15, current_year - 3)
    random_start_month = random.randint(1, 12)
    start_date = datetime.date(random_start_year, random_start_month, 1)
    end_date   = start_date + datetime.timedelta(days=3 * 365)

    n_pairs = len(indicator_lookbacks) * len(z_lens_periods)
    print(f"\n{'─' * 68}")
    print(f"  Test window   : {start_date} → {end_date}  (3 years)")
    print(f"  Search grid   : {len(indicator_lookbacks)} ind_n  ×  "
          f"{len(z_lens_periods)} z_n  =  {n_pairs} pairs")
    print(f"  Features/pair : {len(selected)}  →  "
          f"{len(selected) * 3} lens columns (Z, Z-slope, SOS)")
    print(f"  Total evals   : {iterations * n_pairs}")
    print(f"{'─' * 68}\n")

    all_results = []

    for it in range(1, iterations + 1):
        symbols = random.sample(TITAN_SYMBOLS, min(num_symbols, len(TITAN_SYMBOLS)))
        print(f"\n{'─' * 68}")
        print(f"  Iteration {it}/{iterations}  |  {len(symbols)} symbols")
        print(f"{'─' * 68}")

        for ind_n in indicator_lookbacks:
            for z_n in z_lens_periods:
                # Sanity guard: z_n >> ind_n is physically degenerate
                if z_n > ind_n * 4:
                    print(f"  [SKIP] ind_n={ind_n:3d} | z_n={z_n:3d} — z_n > 4×ind_n")
                    continue

                master_df = load_hybrid_data_parallel(
                    symbols,
                    start_date=start_date,
                    end_date=end_date,
                    indicator_n=ind_n,
                    z_n=z_n,
                    selected_features=selected,
                )

                if master_df.empty:
                    print(f"  [SKIP] ind_n={ind_n:3d} | z_n={z_n:3d} — no data")
                    continue

                scored = _lasso_score(master_df)
                n_syms = master_df['symbol'].nunique() \
                    if 'symbol' in master_df.columns else 0

                row = {
                    'iteration':   it,
                    'indicator_n': ind_n,
                    'z_n':         z_n,
                    'n_symbols':   n_syms,
                    **scored,
                }
                all_results.append(row)

                print(f"  ind_n={ind_n:3d} | z_n={z_n:3d} | syms={n_syms:3d} | "
                      f"R²={scored['r2']:.4f} | "
                      f"Sharpe={scored['sharpe']:+.3f} | "
                      f"nonzero={scored['n_nonzero']:3d} | "
                      f"score={scored['score']:.4f}")

                gc.collect()

    if not all_results:
        print("\n⚠️  No results collected.")
        return pd.DataFrame()

    # ── Aggregate over iterations ──────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    summary = (
        results_df
        .groupby(['indicator_n', 'z_n'])
        .agg(
            persistence = ('iteration',   'count'),
            avg_score   = ('score',        'mean'),
            avg_r2      = ('r2',           'mean'),
            avg_sharpe  = ('sharpe',       'mean'),
            avg_nonzero = ('n_nonzero',    'mean'),
        )
        .reset_index()
        .sort_values('avg_score', ascending=False)
        .reset_index(drop=True)
    )
    summary['rank'] = summary.index + 1

    print(f"\n{'═' * 70}")
    print("  JOINT OPTIMIZATION RESULTS — Ranked by Sovereign Score")
    print(f"  Score = 0.70×Impact + 0.15×Uniqueness + 0.15×Stability")
    print(f"  Features tested: {', '.join(str(_CATALOG_BY_KEY[k][0]) for k in selected)}")
    print(f"{'═' * 70}")
    print(f"  {'RNK':<4} {'ind_n':>6} {'z_n':>5} {'SCORE':>7} "
          f"{'R²':>7} {'Sharpe':>8} {'Nonzero':>8} {'Persist':>8}")
    print(f"  {'─' * 66}")
    for _, row in summary.head(20).iterrows():
        print(f"  {int(row['rank']):02d}.  "
              f"{int(row['indicator_n']):>6} "
              f"{int(row['z_n']):>5} "
              f"{row['avg_score']:>7.4f} "
              f"{row['avg_r2']:>7.4f} "
              f"{row['avg_sharpe']:>+8.3f} "
              f"{row['avg_nonzero']:>8.1f} "
              f"{int(row['persistence']):>5}/{iterations}")

    best = summary.iloc[0]
    print(f"\n{'═' * 70}")
    print(f"  🏆 OPTIMAL PAIR — indicator_n={int(best['indicator_n'])}, "
          f"z_n={int(best['z_n'])}")
    print(f"     Sovereign Score = {best['avg_score']:.4f}")
    print(f"     R²={best['avg_r2']:.4f} | "
          f"Sharpe={best['avg_sharpe']:+.3f} | "
          f"Nonzero={best['avg_nonzero']:.1f}")
    print(f"{'═' * 70}\n")

    return summary


if __name__ == "__main__":
    run_lookback_tester()
