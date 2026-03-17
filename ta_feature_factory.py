# ==============================================================================
# ta_feature_factory.py — Joint Lookback × Z-Lens Optimizer
# Base: Sovereign Titan v4.30 (Ron Harper Edition)
#
# What this adds on top of Ron Harper 4.30:
#   1. Numbered feature catalog — choose which indicators to test each run
#   2. Parameterised generate_factory_features_v2(df, indicator_n, z_n, selected)
#      indicator_n scales ALL indicator windows; z_n is the ONLY lens window
#   3. Exhaustive grid search over the (indicator_n × z_n) joint space
#   4. Multi-horizon evaluation — grid scored across N forward-day targets
#   5. Iteration averaging — each grid point run N times for stable estimates
#
# Mandate (Sovereign framework):
#   Never optimise the seed in isolation. indicator_n and z_n are COUPLED.
#   Grid finds the joint optimum; GA-style filtering is left to the BRAIN_LOCKS.
# ==============================================================================
from __future__ import annotations
import os, gc, warnings, datetime, random, functools, logging
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('peewee').setLevel(logging.CRITICAL)

# ── Colab / local detection ───────────────────────────────────────────────────
try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import tensorflow as tf

# ==============================================================================
# BLOCK 0: GPU SETUP  (Ron Harper 4.30)
# ==============================================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU — running on CPU.")
        return False
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"✅ Mixed precision: {tf.keras.mixed_precision.global_policy().name}")
    with tf.device('/device:GPU:0'):
        _ = tf.random.normal((10, 10)) @ tf.random.normal((10, 10))
    print(f"✅ GPU confirmed: {tf.test.gpu_device_name()}")
    return True

GPU_AVAILABLE = setup_gpu()
DEVICE        = '/device:GPU:0' if GPU_AVAILABLE else '/cpu:0'
print(f"[SYSTEM] Active compute device: {DEVICE}\n")

# ==============================================================================
# BLOCK 1: SYSTEM INITIALISATION  (Ron Harper 4.30)
# ==============================================================================
import numpy as np, pandas as pd, yfinance as yf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import GRU, LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from numba import jit
from datetime import datetime as _dt

# ── Output directory ───────────────────────────────────────────────────────────
if IN_COLAB:
    if not os.path.exists('/content/drive'):
        _colab_drive.mount('/content/drive', force_remount=True)
    _BASE_OUT = '/content/drive/MyDrive/judicial_results'
else:
    _BASE_OUT = os.path.join(os.path.dirname(__file__), 'judicial_results')

TEST_NAME        = "TA_FeatureFactory_JointOpt"
OUTPUT_DIR       = os.path.join(_BASE_OUT, TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
# FEATURE CATALOG  — numbered reference for the interactive per-run selector
# (num, internal_key, display_description, family)
# ==============================================================================
FEATURE_CATALOG: list[tuple] = [
    ( 1, 'er',               'Efficiency Ratio (ER)',                     'Trend'),
    ( 2, 'vidya_cmo',        'VIDYA CMO  (Directional Efficiency)',        'Trend'),
    ( 3, 'r_sq',             'R-squared  (Trend Linearity)',               'Trend'),
    ( 4, 'hurst',            'Hurst Exponent  (Long Memory)',              'Regime'),
    ( 5, 'shannon',          'Shannon Entropy  (Market Noise)',            'Regime'),
    ( 6, 'adx',              'ADX  (Trend Strength)',                      'Trend'),
    ( 7, 'logistic_prob',    'Logistic Prob  (Sigmoid of LR Slope)',       'Momentum'),
    ( 8, 'aroon_up',         'Aroon Up  (Recency of Rolling High)',        'Momentum'),
    ( 9, 'donchian_l',       'Donchian High Long  (Breakout Position)',    'Breakout'),
    (10, 'dispersion',       'Dispersion  (3-MA Spread)',                  'Volatility'),
    (11, 'lr_slope',         'Linear Regression Slope',                    'Momentum'),
    (12, 'tema_s_pct',       'TEMA Short  %   (hlc / TEMA_s − 1)',        'Price-MA'),
    (13, 'tema_m_pct',       'TEMA Primary %  (hlc / TEMA_m − 1)',        'Price-MA'),
    (14, 'sma_xs_pct',       'SMA XShort %    (hlc / SMA_xs − 1)',       'Price-MA'),
    (15, 'sma_m_pct',        'SMA Primary %   (hlc / SMA_m − 1)',        'Price-MA'),
    (16, 'hma_m_pct',        'HMA Primary %   (hlc / HMA_m − 1)',        'Price-MA'),
    (17, 'kalman_pct',       'Kalman %         (hlc / Kalman − 1)',       'Price-MA'),
    (18, 'kalman_sma_ratio', 'Kalman / SMA Ratio',                        'Ratio'),
    (19, 'tema_kal_ratio',   'TEMA / Kalman Ratio',                       'Ratio'),
    (20, 'exhaustion',       'Exhaustion   (hlc − Kalman) / ATR',         'Mean-Rev'),
    (21, 'exhaustion_xl',    'Exhaustion XL  (XL-bar range position)',    'Mean-Rev'),
    (22, 'ratio_acc',        'Slope Acceleration  (slope_s / slope_m)',   'Ratio'),
    (23, 'ratio_snr',        'Signal-to-Noise     (|slope_m| / ATR)',     'Ratio'),
    (24, 'curvature_diff',   'Curvature Diff      (slope_s − slope_xl)', 'Ratio'),
    (25, 'cycle_vs_trend',   'Cycle vs Trend      (|CoG| / R²)',          'Ratio'),
    (26, 'ratio_eff_slope',  'Efficiency × Slope Direction',              'Ratio'),
    (27, 'ratio_pers_slope', 'Persistence × Slope  (Hurst × sign)',       'Ratio'),
    (28, 'ratio_struct',     'Structure Ratio      (TEMA_s − SMA_m)',     'Ratio'),
    (29, 'adx_entropy',      'ADX / Entropy Ratio',                       'Ratio'),
    (30, 'breakout_eff',     'Breakout Efficiency  (Donchian / ER)',      'Ratio'),
    (31, 'r_sq_hurst',       'R² / Hurst Ratio',                         'Ratio'),
    (32, 'vhf_adx',          'VHF / ADX Ratio',                          'Ratio'),
    (33, 'cog',              'Center of Gravity  (Rolling Deviation)',    'Oscillator'),
]

_CAT_BY_NUM = {n: (k, d, f) for n, k, d, f in FEATURE_CATALOG}
_CAT_BY_KEY = {k: (n, d, f) for n, k, d, f in FEATURE_CATALOG}
ALL_KEYS    = [k for _, k, _, _ in FEATURE_CATALOG]

def print_feature_catalog() -> None:
    fams: dict[str, list] = {}
    for num, key, desc, fam in FEATURE_CATALOG:
        fams.setdefault(fam, []).append((num, desc))
    print("\n" + "═" * 68)
    print("  FEATURE CATALOG  — pick the seeds to optimise this run")
    print("═" * 68)
    for fam, items in fams.items():
        print(f"\n  [{fam}]")
        for num, desc in items:
            print(f"    {num:>2}.  {desc}")
    print()
    print("  Syntax:  1,3,5-10,15   |   'all' = every feature")
    print("═" * 68)

def parse_selection(raw: str) -> list[str]:
    raw = raw.strip().lower()
    if raw == 'all':
        return list(ALL_KEYS)
    nums: set[int] = set()
    for tok in raw.split(','):
        tok = tok.strip()
        if '-' in tok:
            lo, hi = tok.split('-')
            nums.update(range(int(lo), int(hi) + 1))
        elif tok.isdigit():
            nums.add(int(tok))
    return [_CAT_BY_NUM[n][0] for n in sorted(nums) if n in _CAT_BY_NUM]

# ==============================================================================
# BLOCK 2: NUMBA JIT ROLLING KERNELS  (Ron Harper 4.30 — unchanged)
# ==============================================================================
@jit(nopython=True, cache=True)
def _lin_slope_nb(y):
    n = len(y)
    if n < 2: return 0.0
    xm = (n - 1) / 2.0
    ym = 0.0
    for i in range(n): ym += y[i]
    ym /= n
    num = den = 0.0
    for i in range(n):
        dx = i - xm
        num += dx * (y[i] - ym)
        den += dx * dx
    return num / den if den != 0.0 else 0.0

@jit(nopython=True, cache=True)
def _rolling_linslope(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _lin_slope_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _hurst_nb(y):
    n = len(y)
    if n < 2: return 0.5
    m = 0.0
    for i in range(n): m += y[i]
    m /= n
    v = 0.0
    for i in range(n): v += (y[i] - m) ** 2
    s = (v / n) ** 0.5
    if s < 1e-12: return 0.5
    r = np.log(s + 1e-9) / np.log(n)
    return r if not np.isnan(r) else 0.5

@jit(nopython=True, cache=True)
def _rolling_hurst(arr, window):
    n = len(arr); out = np.full(n, 0.5)
    for i in range(window - 1, n):
        out[i] = _hurst_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _cog_nb(y):
    n = len(y)
    if n < 2: return 0.0
    num = den = 0.0
    for i in range(n):
        num += (i + 1) * y[i]
        den += y[i]
    return -num / (den + 1e-9)

@jit(nopython=True, cache=True)
def _rolling_cog(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _cog_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _shannon_nb(y, bins=10):
    n = len(y)
    if n < 2: return 0.0
    mn = mx = y[0]
    for i in range(1, n):
        if y[i] < mn: mn = y[i]
        if y[i] > mx: mx = y[i]
    if mx == mn: return 0.0
    counts = np.zeros(bins)
    for i in range(n):
        idx = int((y[i] - mn) / (mx - mn) * bins)
        if idx >= bins: idx = bins - 1
        counts[idx] += 1.0
    e = 0.0
    for i in range(bins):
        p = counts[i] / n + 1e-9
        e -= p * np.log(p)
    return e

@jit(nopython=True, cache=True)
def _rolling_shannon(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _shannon_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _r_sq_nb(y):
    n = len(y)
    if n < 2: return 0.0
    xm = (n - 1) / 2.0
    ym = 0.0
    for i in range(n): ym += y[i]
    ym /= n
    num = dx2 = dy2 = 0.0
    for i in range(n):
        dx = i - xm; dy = y[i] - ym
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy
    if dx2 == 0.0 or dy2 == 0.0: return 0.0
    r = num / ((dx2 ** 0.5) * (dy2 ** 0.5))
    return r * r

@jit(nopython=True, cache=True)
def _rolling_r_sq(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _r_sq_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _rolling_wma(arr, window):
    n = len(arr); out = np.full(n, np.nan)
    ws = window * (window + 1) / 2.0
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += arr[i - window + 1 + j] * (j + 1)
        out[i] = s / ws
    return out

@jit(nopython=True, cache=True)
def _kalman_numba(price, r=0.0001, q=0.001):
    xh = np.zeros_like(price); p = np.zeros_like(price)
    xh[0] = price[0]; p[0] = 1.0
    for t in range(1, len(price)):
        pm    = p[t-1] + q
        k     = pm / (pm + r)
        xh[t] = xh[t-1] + k * (price[t] - xh[t-1])
        p[t]  = (1 - k) * pm
    return xh

def _warm_up_numba():
    d = np.random.randn(60).astype(np.float64)
    _rolling_linslope(d, 10); _rolling_hurst(d, 30); _rolling_cog(d, 20)
    _rolling_shannon(d, 20);  _rolling_r_sq(d, 30);  _rolling_wma(d, 10)
    _kalman_numba(d)
    print("✅ Numba kernels compiled and ready")

_warm_up_numba()

# ==============================================================================
# BLOCK 3: PARAMETERISED FEATURE FACTORY  (Ron Harper 4.30 — extended)
#
# indicator_n  — primary lookback; all indicator windows scale from this:
#     n_s  = max(5,  indicator_n // 2)   short
#     n_m  = max(10, indicator_n)         primary / mid
#     n_l  = max(20, indicator_n * 2)     long
#     n_xl = max(30, indicator_n * 3)     extra-long
#     n_atr= max(7,  indicator_n // 2)    ATR / ADX period
#
# z_n          — the SINGLE lens window replacing Ron Harper's fixed [10, 90].
#                Grid searches for the optimal z_n jointly with indicator_n.
#
# selected_features — list of keys from FEATURE_CATALOG.
#                     All intermediates are always computed; only selected
#                     seeds enter the Z-lens and appear as LENS_* columns.
#
# forward_days — if provided, T_FINAL columns are generated for each horizon:
#                T_FINAL_1d, T_FINAL_3d, etc.  T_FINAL (1d default) always set.
# ==============================================================================
def generate_factory_features_v2(df: pd.DataFrame,
                                  indicator_n: int = 20,
                                  z_n: int = 20,
                                  selected_features: list | None = None,
                                  forward_days: list[int] | None = None,
                                  ) -> pd.DataFrame:
    if selected_features is None:
        selected_features = ALL_KEYS
    if forward_days is None:
        forward_days = [1]
    sel = set(selected_features)
    df  = df.copy()
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

    # ── Multi-horizon targets ─────────────────────────────────────────────────
    for fwd in forward_days:
        col = f'T_FINAL_{fwd}d'
        df[col] = np.where(df['close'].shift(-fwd) > df['close'], 1, 0)
    # Keep T_FINAL (1-day) as the default for backward compatibility
    df['T_FINAL'] = df[f'T_FINAL_{forward_days[0]}d']

    hlc = df['hlc3'].values.astype(np.float64)
    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    cl  = df['close'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    idx = df.index

    # ── Window family ──────────────────────────────────────────────────────────
    n_s   = max(5,  indicator_n // 2)
    n_m   = max(10, indicator_n)
    n_l   = max(20, indicator_n * 2)
    n_xl  = max(30, indicator_n * 3)
    n_atr = max(7,  indicator_n // 2)

    # ── Moving averages ────────────────────────────────────────────────────────
    ema_s  = pd.Series(hlc, index=idx).ewm(span=n_s).mean().values
    ema_s2 = pd.Series(ema_s,  index=idx).ewm(span=n_s).mean().values
    ema_s3 = pd.Series(ema_s2, index=idx).ewm(span=n_s).mean().values
    tema_s = 3*ema_s - 3*ema_s2 + ema_s3
    sma_xs = pd.Series(hlc, index=idx).rolling(max(3, n_s // 2)).mean().values
    sma_m  = pd.Series(hlc, index=idx).rolling(n_m).mean().values
    ema_m  = pd.Series(hlc, index=idx).ewm(span=n_m).mean().values
    ema_m2 = pd.Series(ema_m,  index=idx).ewm(span=n_m).mean().values
    ema_m3 = pd.Series(ema_m2, index=idx).ewm(span=n_m).mean().values
    tema_m = 3*ema_m - 3*ema_m2 + ema_m3
    wma1   = _rolling_wma(hlc, n_s)
    wma2   = _rolling_wma(hlc, n_m)
    hma_m  = pd.Series(2*wma1 - wma2, index=idx).rolling(
                 max(3, int(n_s**0.5))).mean().values
    kalman = _kalman_numba(hlc)

    # ── Efficiency / trend strength ────────────────────────────────────────────
    hlc_s       = pd.Series(hlc, index=idx)
    er_m        = (hlc_s.diff(n_m).abs() /
                   (hlc_s.diff().abs().rolling(n_m).sum() + 1e-9)).values
    vidya_cmo_m = (hlc_s.diff().rolling(n_m).sum() /
                   (hlc_s.diff().abs().rolling(n_m).sum() + 1e-9)).values

    # ── Numba rolling ──────────────────────────────────────────────────────────
    r_sq_m    = _rolling_r_sq(hlc, n_m)
    hurst_l   = _rolling_hurst(hlc, n_l)
    shannon_m = _rolling_shannon(hlc, n_m)
    cog_m     = _rolling_cog(hlc, n_m)

    # ── Linear slopes ──────────────────────────────────────────────────────────
    linreg_s  = _rolling_linslope(hlc, n_s)
    linreg_m  = _rolling_linslope(hlc, n_m)
    linreg_xl = _rolling_linslope(hlc, n_xl)
    slope_std = np.nanstd(linreg_m) + 1e-9
    logistic_m = 1.0 / (1.0 + np.exp(-linreg_m / slope_std))

    # ── MTSI ──────────────────────────────────────────────────────────────────
    tp_v = pd.Series(hlc * vol, index=idx)
    vs   = pd.Series(vol, index=idx)
    cs   = pd.Series(cl,  index=idx)
    _mtsi = (cs - (tp_v.rolling(2).sum() /
                   (vs.rolling(2).sum() + 1e-9))).ewm(span=3).mean().values  # noqa

    # ── ADX ────────────────────────────────────────────────────────────────────
    cl_p = np.roll(cl, 1); cl_p[0] = cl[0]
    tr   = np.maximum(hi-lo, np.maximum(np.abs(hi-cl_p), np.abs(lo-cl_p)))
    atr  = pd.Series(tr, index=idx).rolling(n_atr).mean().values
    hi_p = np.roll(hi, 1); hi_p[0] = hi[0]
    lo_p = np.roll(lo, 1); lo_p[0] = lo[0]
    pdm  = np.where((hi-hi_p)>(lo_p-lo), np.maximum(hi-hi_p,0), 0).astype(np.float64)
    mdm  = np.where((lo_p-lo)>(hi-hi_p), np.maximum(lo_p-lo,0), 0).astype(np.float64)
    pdi  = 100*(pd.Series(pdm,index=idx).rolling(n_atr).mean() /
                (pd.Series(atr,index=idx) + 1e-9))
    mdi  = 100*(pd.Series(mdm,index=idx).rolling(n_atr).mean() /
                (pd.Series(atr,index=idx) + 1e-9))
    adx  = (100*np.abs(pdi-mdi)/(pdi+mdi+1e-9)).rolling(n_atr).mean().values

    # ── Dispersion ─────────────────────────────────────────────────────────────
    sma_l = pd.Series(hlc, index=idx).rolling(n_l).mean().values
    disp  = np.std(np.stack([hlc/(sma_l+1e-9)-1,
                              hlc/(tema_m+1e-9)-1,
                              hlc/(kalman+1e-9)-1], axis=1), axis=1)

    # ── Donchian / Aroon ──────────────────────────────────────────────────────
    hi_s  = pd.Series(hi, index=idx)
    lo_s  = pd.Series(lo, index=idx)
    don_m = (hi_s / hi_s.rolling(n_m).max() - 1).values
    don_l = (hi_s / hi_s.rolling(n_l).max() - 1).values
    aroon = hi_s.rolling(n_m).apply(
        lambda x: float(np.argmax(x)) / len(x), raw=True).values

    # ── Price-MA ratios ────────────────────────────────────────────────────────
    tema_s_pct = hlc/(tema_s+1e-9) - 1
    tema_m_pct = hlc/(tema_m+1e-9) - 1
    sma_xs_pct = hlc/(sma_xs+1e-9) - 1
    sma_m_pct  = hlc/(sma_m +1e-9) - 1
    hma_m_pct  = hlc/(hma_m +1e-9) - 1
    kalman_pct = hlc/(kalman+1e-9) - 1

    # ── VHF ────────────────────────────────────────────────────────────────────
    cda = pd.Series(np.abs(np.diff(cl, prepend=cl[0])), index=idx)
    vhf = ((hi_s.rolling(n_m).max() - lo_s.rolling(n_m).min()) /
           (cda.rolling(n_m).sum() + 1e-9)).values

    # ── Ratio features ─────────────────────────────────────────────────────────
    kal_sma  = kalman/(sma_m+1e-9) - 1
    tema_kal = tema_m/(kalman+1e-9) - 1
    exh      = (hlc - kalman) / (atr + 1e-9)
    hi_xl    = pd.Series(hlc,index=idx).rolling(n_xl).max().values
    lo_xl    = pd.Series(hlc,index=idx).rolling(n_xl).min().values
    exh_xl   = (hi_xl - hlc) / (hi_xl - lo_xl + 1e-9)
    r_acc    = linreg_s / (np.abs(linreg_m)+1e-9) * np.sign(linreg_m)
    r_snr    = np.abs(linreg_m) / (atr+1e-9)
    curv     = linreg_s - linreg_xl
    cyc_tr   = np.abs(cog_m) / (r_sq_m+1e-9)
    r_eff    = er_m * np.sign(linreg_m)
    r_pers   = hurst_l * np.sign(linreg_m)
    r_str    = (tema_s_pct - sma_m_pct) / (np.abs(sma_m_pct)+1e-9)
    adx_ent  = adx / (shannon_m+1e-9)
    bk_eff   = np.abs(don_m) / (er_m+1e-9)
    rsq_h    = r_sq_m / (hurst_l+1e-9)
    vhf_adx  = vhf / (adx+1e-9)

    # ── Full seed dict ─────────────────────────────────────────────────────────
    SEEDS: dict[str, pd.Series] = {
        'er':               pd.Series(er_m,       index=idx),
        'vidya_cmo':        pd.Series(vidya_cmo_m, index=idx),
        'r_sq':             pd.Series(r_sq_m,      index=idx),
        'hurst':            pd.Series(hurst_l,     index=idx),
        'shannon':          pd.Series(shannon_m,   index=idx),
        'adx':              pd.Series(adx,         index=idx),
        'logistic_prob':    pd.Series(logistic_m,  index=idx),
        'aroon_up':         pd.Series(aroon,       index=idx),
        'donchian_l':       pd.Series(don_l,       index=idx),
        'dispersion':       pd.Series(disp,        index=idx),
        'lr_slope':         pd.Series(linreg_m,    index=idx),
        'tema_s_pct':       pd.Series(tema_s_pct,  index=idx),
        'tema_m_pct':       pd.Series(tema_m_pct,  index=idx),
        'sma_xs_pct':       pd.Series(sma_xs_pct,  index=idx),
        'sma_m_pct':        pd.Series(sma_m_pct,   index=idx),
        'hma_m_pct':        pd.Series(hma_m_pct,   index=idx),
        'kalman_pct':       pd.Series(kalman_pct,  index=idx),
        'kalman_sma_ratio': pd.Series(kal_sma,     index=idx),
        'tema_kal_ratio':   pd.Series(tema_kal,    index=idx),
        'exhaustion':       pd.Series(exh,         index=idx),
        'exhaustion_xl':    pd.Series(exh_xl,      index=idx),
        'ratio_acc':        pd.Series(r_acc,       index=idx),
        'ratio_snr':        pd.Series(r_snr,       index=idx),
        'curvature_diff':   pd.Series(curv,        index=idx),
        'cycle_vs_trend':   pd.Series(cyc_tr,      index=idx),
        'ratio_eff_slope':  pd.Series(r_eff,       index=idx),
        'ratio_pers_slope': pd.Series(r_pers,      index=idx),
        'ratio_struct':     pd.Series(r_str,       index=idx),
        'adx_entropy':      pd.Series(adx_ent,     index=idx),
        'breakout_eff':     pd.Series(bk_eff,      index=idx),
        'r_sq_hurst':       pd.Series(rsq_h,       index=idx),
        'vhf_adx':          pd.Series(vhf_adx,     index=idx),
    }

    # ── Apply Z-lens ONLY to selected features ────────────────────────────────
    # Physics Trio per seed:
    #   LENS_{z_n}_{key}_z        → Position   (where relative to history)
    #   LENS_{z_n}_{key}_z_slope  → Velocity   (rate of change)
    #   LENS_{z_n}_{key}_z_sos    → Acceleration / SOS  (turning-point signal)
    for name, series in SEEDS.items():
        if name not in sel:
            continue
        arr  = series.values.astype(np.float64)
        rm   = pd.Series(arr, index=idx).rolling(z_n).mean().values
        rs   = pd.Series(arr, index=idx).rolling(z_n).std().values
        z    = (arr - rm) / (rs + 1e-9)
        zs   = _rolling_linslope(z,  z_n)
        zsos = _rolling_linslope(zs, z_n)
        df[f'LENS_{z_n}_{name}_z']       = z
        df[f'LENS_{z_n}_{name}_z_slope'] = zs
        df[f'LENS_{z_n}_{name}_z_sos']   = zsos

    # ── CoG rolling-pct group (only when 'cog' selected) ─────────────────────
    if 'cog' in sel:
        cog_arr = cog_m.astype(np.float64)
        for win in [n_s, n_m]:
            rm = pd.Series(cog_arr, index=idx).rolling(win).mean().values
            df[f'WIN_{win}_cog_pct'] = cog_arr / (rm + 1e-9) - 1

    return (df.replace([np.inf, -np.inf], np.nan)
              .ffill()
              .dropna(subset=['T_FINAL'])
              .fillna(0))

# ==============================================================================
# BLOCK 4: PARALLEL DATA LOADER  (Ron Harper 4.30 — extended with date range)
# ==============================================================================
def _process_symbol_worker(args):
    """Top-level for ProcessPoolExecutor (must be picklable)."""
    symbol, raw_dict, indicator_n, z_n, selected_features, forward_days = args
    try:
        raw_df = pd.DataFrame(raw_dict)
        raw_df.index = pd.to_datetime(raw_df.index)
        if len(raw_df) < max(120, indicator_n * 4):
            return None
        proc = generate_factory_features_v2(
            raw_df, indicator_n, z_n, selected_features, forward_days
        )
        if proc.empty:
            return None
        proc['symbol'] = symbol
        return proc.reset_index()
    except Exception:
        return None

def fetch_data(symbol: str,
               start: str | None = None,
               end:   str | None = None) -> pd.DataFrame | None:
    try:
        kw = dict(interval='1d', progress=False)
        if start and end:
            kw['start'] = str(start)
            kw['end']   = str(end)
        else:
            kw['period'] = '2y'
        data = yf.download(symbol, **kw)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).lower() for c in data.columns]
        return data if 'close' in data.columns else None
    except Exception:
        return None

def load_hybrid_data_parallel(brain_name: str,
                               symbol_list: list,
                               indicator_n: int = 20,
                               z_n:         int = 20,
                               selected_features: list | None = None,
                               forward_days: list[int] | None = None,
                               start_date = None,
                               end_date   = None,
                               dl_workers: int = 20) -> pd.DataFrame:
    if selected_features is None:
        selected_features = ALL_KEYS
    if forward_days is None:
        forward_days = [1]

    raw: dict = {}
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        fmap = {pool.submit(fetch_data, sym, start_date, end_date): sym
                for sym in symbol_list}
        for fut in as_completed(fmap):
            sym  = fmap[fut]
            data = fut.result()
            min_bars = max(120, indicator_n * 4)
            if data is not None and len(data) >= min_bars:
                raw[sym] = data

    if not raw:
        return pd.DataFrame()

    items = [(sym, df.to_dict(), indicator_n, z_n, selected_features, forward_days)
             for sym, df in raw.items()]
    all_data: list = []
    try:
        with ProcessPoolExecutor(max_workers=N_FEATURE_WORKERS) as pool:
            futs = {pool.submit(_process_symbol_worker, item): item[0]
                    for item in items}
            for fut in as_completed(futs):
                res = fut.result()
                if res is not None:
                    all_data.append(res.set_index(res.columns[0]))
    except Exception as e:
        print(f"  ⚠️  ProcessPool failed ({e}) — sequential fallback")
        for item in items:
            res = _process_symbol_worker(item)
            if res is not None:
                all_data.append(res.set_index(res.columns[0]))

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, axis=0)

# ==============================================================================
# BLOCK 5: GPU-ACCELERATED AUDIT  (Ron Harper 4.30)
# ==============================================================================
def build_full_model(model_type, n_features, seq_len, device=DEVICE):
    with tf.device(device):
        model = Sequential([
            Input(shape=(seq_len, n_features)),
            GRU(128, return_sequences=True) if model_type == 'GRU'
                else LSTM(128, return_sequences=True),
            Dropout(0.2),
            GRU(64) if model_type == 'GRU' else LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1,  activation='sigmoid', dtype='float32'),
        ])
        model.compile(optimizer=Adam(1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

@tf.function
def _eval_accuracy(model, X_b, y_b):
    preds = tf.squeeze(model(X_b, training=False), axis=-1)
    ok    = tf.equal(tf.cast(preds >= 0.5, tf.int32), tf.cast(y_b, tf.int32))
    return tf.reduce_mean(tf.cast(ok, tf.float32))

def run_judicial_audit(brain_name, master_df, model_type='GRU',
                       seq_len=10, epochs=30, batch_size=2048,
                       target_col: str = 'T_FINAL',
                       ) -> tuple[pd.DataFrame, float]:
    """
    Returns
    -------
    report_df     : permutation-importance DataFrame  (Feature, I_raw)
    baseline_acc  : scalar validation accuracy
    """
    feat_cols = [c for c in master_df.columns
                 if c.startswith('LENS_') or c.startswith('WIN_')]
    n_features = len(feat_cols)
    if n_features == 0:
        return pd.DataFrame(columns=['Feature', 'I_raw']), 0.0

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(master_df[feat_cols].values).astype(np.float32)
    y_raw    = master_df[target_col].values.astype(np.float32)

    n      = len(X_scaled)
    X_seqs = np.stack([X_scaled[i - seq_len:i] for i in range(seq_len, n)])
    y_seqs = y_raw[seq_len:]
    split  = int(len(X_seqs) * 0.8)
    X_tr, X_val = X_seqs[:split], X_seqs[split:]
    y_tr, y_val = y_seqs[:split], y_seqs[split:]

    print(f"  [DATA] train={len(X_tr):,}  val={len(X_val):,}  "
          f"features={n_features}  target={target_col}")

    AUTO  = tf.data.AUTOTUNE
    tr_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
             .shuffle(min(20_000, len(X_tr)), reshuffle_each_iteration=True)
             .batch(batch_size).prefetch(AUTO))
    va_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
             .batch(batch_size * 2).prefetch(AUTO))

    model = build_full_model(model_type, n_features, seq_len)
    with tf.device(DEVICE):
        model.fit(tr_ds, validation_data=va_ds, epochs=epochs,
                  callbacks=[
                      EarlyStopping(monitor='val_loss', patience=5,
                                    restore_best_weights=True),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=3, min_lr=1e-5),
                  ], verbose=1)

    Xvt = tf.constant(X_val)
    yvt = tf.constant(y_val)
    baseline_acc = float(_eval_accuracy(model, Xvt, yvt).numpy())
    print(f"  [MODEL] Val accuracy ({target_col}): {baseline_acc:.4f}")

    rows = []
    for fi, fname in enumerate(tqdm(feat_cols, desc="Permutation scoring")):
        try:
            Xp = X_val.copy()
            flat = Xp[:, :, fi].flatten()
            np.random.shuffle(flat)
            Xp[:, :, fi] = flat.reshape(Xp[:, :, fi].shape)
            pa = float(_eval_accuracy(model, tf.constant(Xp), yvt).numpy())
            rows.append({'Feature': fname, 'I_raw': max(0.0, baseline_acc - pa)})
        except Exception:
            rows.append({'Feature': fname, 'I_raw': 0.0})

    del model; gc.collect(); tf.keras.backend.clear_session()
    return pd.DataFrame(rows), baseline_acc

# ==============================================================================
# BLOCK 6: SOVEREIGN HUNT & DIVERSITY ANCHORS  (Ron Harper 4.30 — unchanged)
# ==============================================================================
BRAIN_LOCKS: dict[str, list] = {'DIRECTION': [], 'EASE': [], 'EXP': []}

def _parse_feature_name(f):
    XFORM = {'z', 'slope', 'sos', 'pct'}
    if f.startswith('LENS_') or f.startswith('WIN_'):
        parts = f.split('_')
        prefix, window = parts[0], parts[1]
        rem = list(parts[2:])
        while rem and rem[-1] in XFORM:
            rem.pop()
        indicator = '_'.join(rem)
        family    = '_'.join(p for p in rem if not p.isdigit())
        return prefix, window, f'{prefix}_{window}_{indicator}', family
    return None, None, f, f

def apply_sovereign_hunt(ledger_df, master_data_df, brain_name, max_slots=19):
    locked = BRAIN_LOCKS.get(brain_name, [])
    cands  = ledger_df.sort_values('I_raw', ascending=False)
    picked = [f for f in locked if f in ledger_df['Feature'].values]
    for lf in locked:
        if lf not in ledger_df['Feature'].values:
            print(f"  ⚠️  BRAIN_LOCK '{lf}' not found")

    CORR_THR = 0.85
    fam_lbw: dict = {}
    for f in picked:
        _, _, lbw, fam = _parse_feature_name(f)
        fam_lbw.setdefault(fam, lbw)

    feat_cols   = [c for c in master_data_df.columns
                   if c.startswith('LENS_') or c.startswith('WIN_')]
    corr_matrix = master_data_df[feat_cols].corr()

    for _, row in cands.iterrows():
        if len(picked) >= max_slots: break
        fn = row['Feature']
        if fn in picked: continue
        _, _, fl, ff = _parse_feature_name(fn)
        if ff in fam_lbw and fam_lbw[ff] != fl: continue
        if picked and corr_matrix[fn].loc[picked].max() > CORR_THR: continue
        picked.append(fn)
        fam_lbw.setdefault(ff, fl)

    pca = PCA()
    pca.fit(RobustScaler().fit_transform(master_data_df[picked]))
    return picked, np.cumsum(pca.explained_variance_ratio_)

def generate_judicial_ledger(brain_name, report_df, master_data_df, iteration=1):
    df = report_df.copy()
    df['I_Norm'] = ((df['I_raw'] - df['I_raw'].min()) /
                    (df['I_raw'].max() - df['I_raw'].min() + 1e-9))
    picks, var_map = apply_sovereign_hunt(df, master_data_df, brain_name)
    csub  = master_data_df[picks].corr().abs()
    avg_c = ((csub.sum().sum() - len(picks)) /
             (len(picks)**2 - len(picks) + 1e-9))

    print(f"\n╔══ {brain_name} SOVEREIGN CORE V4.30 (Iter {iteration}) ══╗")
    print(f"║ {'RNK':<3} | {'FEATURE':<35} | {'UV%':<4} | {'mR':<4} | {'IMPACT':<8} ║")
    print("╠" + "═"*4 + "╬" + "═"*37 + "╬" + "═"*6 + "╬" + "═"*6 + "╬" + "═"*10 + "╣")
    for i, fn in enumerate(picks):
        frow   = df[df['Feature'] == fn].iloc[0]
        locked = fn in BRAIN_LOCKS.get(brain_name, [])
        others = [p for p in picks if p != fn]
        max_r  = csub[fn].loc[others].max() if others else 0.0
        uv     = (1 - csub[fn].loc[others].mean()) * 100 if others else 100.0
        icon   = "🔒" if locked else "🔭"
        print(f"║ {i+1:02d}  | {icon} {fn[:33]:<33} | "
              f"{uv:>3.0f}% | {max_r:.2f} | {frow['I_Norm']:.4f} ║")
        df.loc[df['Feature'] == fn, ['UV%','Max_R','Is_Locked']] = [uv, max_r, locked]

    tv = var_map[-1] if len(var_map) > 0 else 0
    print("╠" + "═"*73 + "╣")
    print(f"║ PCA VARIANCE RETAINED:  {tv*100:>39.2f}% ║")
    print(f"║ AVG CROSS-CORRELATION:  {avg_c:>41.3f} ║")
    print(f"║ SLOTS FILLED:           {len(picks):>41}/19 ║")
    print("╚" + "═"*73 + "╝")
    return df[df['Feature'].isin(picks)]

# ==============================================================================
# GRID SEARCH OBJECTIVE
# ==============================================================================
def _evaluate_pair(ind_n: int, z_n: int,
                   symbols: list,
                   start_date, end_date,
                   selected_features: list,
                   brain_name: str,
                   model_type: str,
                   forward_days: list[int],
                   ) -> float:
    """
    Load data once, evaluate across all forward-day horizons, return mean accuracy.
    """
    master_df = load_hybrid_data_parallel(
        brain_name, symbols,
        indicator_n=ind_n, z_n=z_n,
        selected_features=selected_features,
        forward_days=forward_days,
        start_date=start_date, end_date=end_date,
    )
    if master_df.empty or len(master_df) < 500:
        return 0.0

    horizon_accs: list[float] = []
    for fwd in forward_days:
        target_col = f'T_FINAL_{fwd}d'
        if target_col not in master_df.columns:
            continue
        _, acc = run_judicial_audit(
            brain_name, master_df,
            model_type=model_type,
            target_col=target_col,
        )
        horizon_accs.append(acc)
        print(f"    fwd={fwd}d  acc={acc:.4f}")

    return float(np.mean(horizon_accs)) if horizon_accs else 0.0

# ==============================================================================
# ENTRY POINT
# ==============================================================================
def run_lookback_tester():
    print("\n" + "═"*68)
    print("  TA Feature Factory — Joint Lookback × Z-Lens Optimizer")
    print("  Engine: Exhaustive Grid over user-specified lookback lists")
    print("  Mandate: indicator_n and z_n optimised SIMULTANEOUSLY")
    print("═"*68)

    # ── Feature selection ──────────────────────────────────────────────────────
    print_feature_catalog()
    raw_sel  = input("Select features to optimise this run: ").strip()
    selected = parse_selection(raw_sel)
    if not selected:
        print("⚠️  No valid features selected. Exiting."); return pd.DataFrame()

    print(f"\n  Selected {len(selected)} feature(s):")
    for key in selected:
        n, desc, fam = _CAT_BY_KEY[key]
        print(f"    {n:>2}. [{fam:<10}] {desc}")

    # ── Forward-day targets ────────────────────────────────────────────────────
    print()
    raw_fwd     = input("Forward days to score (e.g. 1,3,5,10): ").strip()
    forward_days = [int(x.strip()) for x in raw_fwd.split(',')
                    if x.strip().isdigit() and int(x.strip()) > 0]
    if not forward_days:
        forward_days = [1]
        print("  (no valid values — defaulting to 1 day)")

    # ── Iterations per grid point ──────────────────────────────────────────────
    raw_iters   = input("Iterations per combo before deciding [default 1]: ").strip()
    n_iterations = int(raw_iters) if raw_iters.isdigit() and int(raw_iters) >= 1 else 1

    # ── Lookback lists ─────────────────────────────────────────────────────────
    print()
    raw_ind = input("indicator_n values to test (e.g. 5,10,20,30,40,60): ").strip()
    raw_z   = input("z_n values to test         (e.g. 5,10,20,30,40,60): ").strip()

    def _parse_csv_ints(s: str) -> list[int]:
        out = []
        for tok in s.split(','):
            tok = tok.strip()
            if tok.isdigit():
                out.append(int(tok))
        return sorted(set(out))

    ind_list = _parse_csv_ints(raw_ind)
    z_list   = _parse_csv_ints(raw_z)
    if not ind_list or not z_list:
        print("⚠️  No valid values parsed. Exiting."); return pd.DataFrame()

    # ── Other inputs ───────────────────────────────────────────────────────────
    brain_ch    = input("Brain (1:DIR / 2:EXP) [default 1]: ").strip() or "1"
    brain_name  = {'1': 'DIRECTION', '2': 'EXP'}.get(brain_ch, 'DIRECTION')
    model_type  = 'GRU' if brain_name == 'DIRECTION' else 'LSTM'
    num_symbols = int(input("Symbols per eval (e.g. 30): ") or "30")

    # ── Random 3-year window within the last 15 years ─────────────────────────
    cy         = 2026
    sy         = random.randint(cy - 15, cy - 3)
    sm         = random.randint(1, 12)
    start_date = datetime.date(sy, sm, 1)
    end_date   = start_date + datetime.timedelta(days=3 * 365)

    import itertools
    grid = list(itertools.product(ind_list, z_list))

    print(f"\n{'─'*68}")
    print(f"  Test window    : {start_date} → {end_date}  (3 years)")
    print(f"  Brain          : {brain_name}  ({model_type})")
    print(f"  indicator_n    : {ind_list}")
    print(f"  z_n            : {z_list}")
    print(f"  Combinations   : {len(grid)}  ({len(ind_list)} × {len(z_list)})")
    print(f"  Forward days   : {forward_days}")
    print(f"  Iterations/combo: {n_iterations}")
    print(f"  Features       : {len(selected)}  →  {len(selected)*3} LENS columns per pair")
    print(f"{'─'*68}\n")

    # ── Exhaustive grid ────────────────────────────────────────────────────────
    best_acc, best_ind_n, best_z_n = -1.0, ind_list[0], z_list[0]
    results: list[tuple] = []   # (ind_n, z_n, mean_acc, [iter_accs])

    for combo_i, (ind_n, z_n) in enumerate(grid, 1):
        print(f"\n[{combo_i}/{len(grid)}] ind_n={ind_n}  z_n={z_n} — "
              f"{n_iterations} iteration(s) × {len(forward_days)} horizon(s)")

        iter_accs: list[float] = []
        for it in range(1, n_iterations + 1):
            # Fresh random symbol draw each iteration for independent estimates
            symbols = random.sample(TITAN_SYMBOLS, min(num_symbols, len(TITAN_SYMBOLS)))
            print(f"  iter {it}/{n_iterations}  symbols={len(symbols)}")
            acc = _evaluate_pair(
                ind_n, z_n,
                symbols, start_date, end_date,
                selected, brain_name, model_type, forward_days,
            )
            iter_accs.append(acc)
            print(f"  → iter {it} mean-horizon acc = {acc:.4f}")

        mean_acc = float(np.mean(iter_accs))
        std_acc  = float(np.std(iter_accs))
        results.append((ind_n, z_n, mean_acc, iter_accs))
        print(f"  ★ combo mean={mean_acc:.4f}  std={std_acc:.4f}")

        if mean_acc > best_acc:
            best_acc, best_ind_n, best_z_n = mean_acc, ind_n, z_n

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  GRID SEARCH — FULL RESULTS")
    iter_header = "  " + "  ".join([f"it{i+1:02d}" for i in range(n_iterations)])
    print(f"  {'#':>3}  {'ind_n':>6}  {'z_n':>5}  {'mean_acc':>9}  {'std':>6}"
          + (f"  {iter_header}" if n_iterations > 1 else ""))
    print(f"  {'─'*55}")
    for i, (ind_n, z_n, mean_acc, iter_accs) in enumerate(results, 1):
        marker   = " ← best" if (ind_n == best_ind_n and z_n == best_z_n) else ""
        std_acc  = float(np.std(iter_accs))
        iter_str = ("  " + "  ".join([f"{a:.4f}" for a in iter_accs])
                    if n_iterations > 1 else "")
        print(f"  {i:>3}  {ind_n:>6}  {z_n:>5}  {mean_acc:>9.4f}  {std_acc:>6.4f}"
              f"{iter_str}{marker}")

    print(f"\n{'═'*68}")
    print(f"  🏆 OPTIMAL PAIR: indicator_n={best_ind_n}  z_n={best_z_n}")
    print(f"     Mean validation accuracy = {best_acc:.4f}  "
          f"(across {n_iterations} iter × {len(forward_days)} horizon(s))")
    print(f"{'═'*68}\n")

    # ── Save results summary ───────────────────────────────────────────────────
    rows = [{'ind_n': r[0], 'z_n': r[1], 'mean_acc': r[2],
             **{f'iter_{j+1}': r[3][j] for j in range(len(r[3]))}}
            for r in results]
    summary_df = pd.DataFrame(rows)
    fname = (f"GridResults_{brain_name}_"
             f"fwd{'_'.join(str(d) for d in forward_days)}_"
             f"{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv")
    fpath = os.path.join(OUTPUT_DIR, fname)
    summary_df.to_csv(fpath, index=False)
    print(f"📂 Grid results saved → {fpath}")

    return summary_df


if __name__ == "__main__":
    run_lookback_tester()
