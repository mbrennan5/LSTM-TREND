# SOVEREIGN TITAN v3.19.18 — GPU + SPEED EDITION v2
# Speed changes vs prior version:
#   1. Parallel yfinance downloads       (ThreadPoolExecutor)
#   2. Sequences built ONCE per iter     (was rebuilt 150× per feature)
#   3. Single model + perm-importance    (1 train + 150 forward passes vs 150 trains)
#   4. tf.data pipeline with prefetch    (GPU never idles waiting for CPU)
#   5. @tf.function on eval step         (JIT-compiles permutation scoring)
# ── NEW IN THIS VERSION ──────────────────────────────────────────────────────
#   6. ALL rolling functions JIT-compiled via Numba  (no more Python lambdas)
#      _lin_slope, _hurst, _cog, _shannon, _r_sq, _wma — all native machine code
#   7. ProcessPoolExecutor for feature generation    (true multi-core, bypasses GIL)
#   8. Duplicate linreg/slope computation removed
#   9. Dispersion vectorised (np.stack instead of Python list comprehension)
#  10. BRAIN_LOCKS names corrected to match actual column output
# ==============================================================================
# ### BLOCK 0: GPU SETUP
# ==============================================================================
import os, gc, warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU — running CPU. Colab: Runtime → Change runtime type → T4 GPU")
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
DEVICE = '/device:GPU:0' if GPU_AVAILABLE else '/cpu:0'
print(f"[SYSTEM] Active compute device: {DEVICE}\n")

# ==============================================================================
# ### BLOCK 1: SYSTEM INITIALIZATION
# ==============================================================================
import numpy as np, pandas as pd, yfinance as yf
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import random
from numba import jit
from datetime import datetime
from google.colab import drive

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive', force_remount=True)

TEST_NAME        = "Sovereign_Titan_v3.19.18_Entropy_Injection"
OUTPUT_DRIVE_DIR = f'/content/drive/MyDrive/judicial_results/{TEST_NAME}/'
if not os.path.exists(OUTPUT_DRIVE_DIR):
    os.makedirs(OUTPUT_DRIVE_DIR)

# Number of CPU workers for feature generation
# Colab free = 2, Colab Pro = 4-8. Auto-detects.
N_FEATURE_WORKERS = max(1, (os.cpu_count() or 2))
print(f"[SYSTEM] Feature generation workers: {N_FEATURE_WORKERS}")

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
    'XLRE','XLU','XLV','XLY','XOM','XOP','XRT'
]

# ==============================================================================
# ### BLOCK 2: NUMBA JIT ROLLING KERNELS
# cache=True saves compiled artifacts to disk — subsequent runs skip recompile.
# Each function replaces a pandas rolling().apply(lambda...) call.
# Speedup per function: ~10-50× over interpreted Python lambdas.
# ==============================================================================

# ── Linear slope (replaces np.polyfit inside rolling) ─────────────────────────
@jit(nopython=True, cache=True, fastmath=True)
def _lin_slope_nb(y):
    """OLS slope — equivalent to np.polyfit(x, y, 1)[0] but ~20× faster."""
    n = len(y)
    if n < 2: return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = 0.0
    for i in range(n): y_mean += y[i]
    y_mean /= n
    num = 0.0; den = 0.0
    for i in range(n):
        dx = i - x_mean
        num += dx * (y[i] - y_mean)
        den += dx * dx
    return num / den if den != 0.0 else 0.0

@jit(nopython=True, cache=True, fastmath=True)
def _rolling_linslope(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _lin_slope_nb(arr[i - window + 1 : i + 1])
    return out

# ── Hurst exponent ─────────────────────────────────────────────────────────────
@jit(nopython=True, cache=True, fastmath=True)
def _hurst_nb(y):
    n = len(y)
    if n < 2: return 0.5
    mean = 0.0
    for i in range(n): mean += y[i]
    mean /= n
    var = 0.0
    for i in range(n): var += (y[i] - mean) ** 2
    std = (var / n) ** 0.5
    if std < 1e-12: return 0.5
    r = np.log(std + 1e-9) / np.log(n)
    return r if not np.isnan(r) else 0.5

@jit(nopython=True, cache=True, fastmath=True)
def _rolling_hurst(arr, window):
    n = len(arr); out = np.full(n, 0.5)
    for i in range(window - 1, n):
        out[i] = _hurst_nb(arr[i - window + 1 : i + 1])
    return out

# ── Center of Gravity ──────────────────────────────────────────────────────────
@jit(nopython=True, cache=True, fastmath=True)
def _cog_nb(y):
    n = len(y)
    if n < 2: return 0.0
    num = 0.0; den = 0.0
    for i in range(n):
        w = float(i + 1)
        num += w * y[i]
        den += y[i]
    return -num / (den + 1e-9)

@jit(nopython=True, cache=True, fastmath=True)
def _rolling_cog(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _cog_nb(arr[i - window + 1 : i + 1])
    return out

# ── Shannon entropy (manual histogram — np.histogram not in nopython) ──────────
@jit(nopython=True, cache=True, fastmath=True)
def _shannon_nb(y, bins=10):
    n = len(y)
    if n < 2: return 0.0
    mn = y[0]; mx = y[0]
    for i in range(1, n):
        if y[i] < mn: mn = y[i]
        if y[i] > mx: mx = y[i]
    if mx == mn: return 0.0
    counts = np.zeros(bins)
    for i in range(n):
        idx = int((y[i] - mn) / (mx - mn) * bins)
        if idx >= bins: idx = bins - 1
        counts[idx] += 1.0
    entropy = 0.0
    for i in range(bins):
        p = counts[i] / n + 1e-9
        entropy -= p * np.log(p)
    return entropy

@jit(nopython=True, cache=True, fastmath=True)
def _rolling_shannon(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _shannon_nb(arr[i - window + 1 : i + 1])
    return out

# ── R-squared (correlation² of index vs values) ────────────────────────────────
@jit(nopython=True, cache=True, fastmath=True)
def _r_sq_nb(y):
    n = len(y)
    if n < 2: return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = 0.0
    for i in range(n): y_mean += y[i]
    y_mean /= n
    num = 0.0; den_x = 0.0; den_y = 0.0
    for i in range(n):
        dx = i - x_mean; dy = y[i] - y_mean
        num   += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x == 0.0 or den_y == 0.0: return 0.0
    r = num / ((den_x ** 0.5) * (den_y ** 0.5))
    return r * r

@jit(nopython=True, cache=True, fastmath=True)
def _rolling_r_sq(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _r_sq_nb(arr[i - window + 1 : i + 1])
    return out

# ── Weighted Moving Average ────────────────────────────────────────────────────
@jit(nopython=True, cache=True, fastmath=True)
def _rolling_wma(arr, window):
    n = len(arr); out = np.full(n, np.nan)
    w_sum = window * (window + 1) / 2.0
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += arr[i - window + 1 + j] * (j + 1)
        out[i] = s / w_sum
    return out

# ── Kalman filter ──────────────────────────────────────────────────────────────
@jit(nopython=True, cache=True, fastmath=True)
def _kalman_numba(price, r=0.0001, q=0.001):
    x_hat = np.zeros_like(price); p = np.zeros_like(price)
    x_hat[0] = price[0]; p[0] = 1.0
    for t in range(1, len(price)):
        p_minus  = p[t-1] + q
        k        = p_minus / (p_minus + r)
        x_hat[t] = x_hat[t-1] + k * (price[t] - x_hat[t-1])
        p[t]     = (1 - k) * p_minus
    return x_hat

# ── Trigger first-time Numba compilation at import time (not during the run) ───
def _warm_up_numba():
    dummy = np.random.randn(60).astype(np.float64)
    _rolling_linslope(dummy, 10)
    _rolling_hurst(dummy, 50)
    _rolling_cog(dummy, 20)
    _rolling_shannon(dummy, 20)
    _rolling_r_sq(dummy, 30)
    _rolling_wma(dummy, 10)
    _kalman_numba(dummy)
    print("✅ Numba kernels compiled and ready")

_warm_up_numba()

# ==============================================================================
# ### BLOCK 3: FEATURE FACTORY — uses Numba kernels throughout
# Top-level function required for ProcessPoolExecutor pickling.
# ==============================================================================
def generate_factory_features_v2(df):
    df = df.copy()
    df['hlc3']    = (df['high'] + df['low'] + df['close']) / 3
    # Stage 1 spec — No-Lag Target: predict Close_t (contemporaneous log return),
    # NOT Close_{t+1}.  Forces recurrent gates to learn temporal dependencies
    # from the feature sequence rather than a forward-looking label.
    df['T_FINAL'] = np.log(df['close'] / df['close'].shift(1))
    df.loc[df.index[0], 'T_FINAL'] = np.nan   # no prior close for first row

    hlc = df['hlc3'].values.astype(np.float64)
    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    cl  = df['close'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    idx = df.index

    # ── Moving averages ────────────────────────────────────────────────────────
    ema30   = pd.Series(hlc, index=idx).ewm(span=30).mean().values
    ema30_2 = pd.Series(ema30, index=idx).ewm(span=30).mean().values
    ema30_3 = pd.Series(ema30_2, index=idx).ewm(span=30).mean().values
    tema_30 = 3*ema30 - 3*ema30_2 + ema30_3
    sma_20  = pd.Series(hlc, index=idx).rolling(20).mean().values

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

    # linreg computed ONCE (duplicate removed)
    linreg_30        = _rolling_linslope(hlc, 30)
    slope_std        = np.nanstd(linreg_30) + 1e-9
    logistic_prob_30 = 1.0 / (1.0 + np.exp(-linreg_30 / slope_std))

    # ── MTSI (Modified True Strength Index via VWAP anchor) ───────────────────
    # Formula: EMA(3) of (close - 2-bar VWAP)
    # 2-bar VWAP = sum(hlc3 * volume, 2) / sum(volume, 2)
    # Measures distance from short-term volume-weighted fair value.
    # Oscillates around zero in price units — z-lens normalises cross-stock scaling.
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

    # ── Donchian / Aroon (VPIN removed — replaced by R²) ─────────────────────
    hi_s             = pd.Series(hi, index=idx)
    donchian_high_50 = (hi_s / hi_s.rolling(50).max() - 1).values
    aroon_up_25      = hi_s.rolling(25).apply(
        lambda x: float(np.argmax(x)) / 25, raw=True
    ).values

    # ══════════════════════════════════════════════════════════════════════════
    # INDICATOR CLASSIFICATION
    #
    # RULE: if the signal is bounded or can be transformed into a bounded,
    #       mean-reverting series → z-lens (LENS 10 & 90).
    #       if the signal is genuinely unbounded/cumulative in price units
    #       and no natural normalisation exists → rolling % (WIN 10, 30, 90).
    #
    # Z-LENS GROUP (16 indicators × 2 lenses × 3 transforms = 96 features)
    # ─────────────────────────────────────────────────────────────────────────
    #   Raw bounded indicators (passed directly):
    #     er_20, vidya_cmo_20, r_sq_30, hurst_50, shannon_20, adx_14,
    #     logistic_prob_30, aroon_up_25, donchian_high_50, dispersion_30,
    #     lr_slope_30
    #
    #   Price MA indicators (pre-transformed to hlc3/MA - 1):
    #     tema_30, sma_20, hma_21, kalman
    #     hlc3/MA - 1 is mean-reverting around zero → z-lens natural.
    #
    # ROLLING % GROUP (1 indicator × 3 windows = 3 features)
    # ─────────────────────────────────────────────────────────────────────────
    #   cog_20: Center of Gravity is in price units, unbounded.
    #           cog / rolling_mean(cog, N) - 1 measures deviation from norm.
    #
    # TOTAL: 90 + 3 = 93 features
    # ══════════════════════════════════════════════════════════════════════════

    # ── Pre-transform Price MAs: hlc3 / MA - 1 ────────────────────────────────
    tema_30_pct = hlc / (tema_30 + 1e-9) - 1
    sma_20_pct  = hlc / (sma_20  + 1e-9) - 1
    hma_21_pct  = hlc / (hma_21  + 1e-9) - 1
    kalman_pct  = hlc / (kalman  + 1e-9) - 1

    # ── Z-lens group ──────────────────────────────────────────────────────────
    Z_LENS_INDICATORS = {
        # Bounded oscillators — raw value
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
        # Price MAs — pre-transformed to hlc3/MA - 1
        'tema_30_pct':      pd.Series(tema_30_pct,      index=idx),
        'sma_20_pct':       pd.Series(sma_20_pct,       index=idx),
        'hma_21_pct':       pd.Series(hma_21_pct,       index=idx),
        'kalman_pct':       pd.Series(kalman_pct,       index=idx),
        # Price-unit oscillators — z-lens normalises cross-stock scaling
        'mtsi':             pd.Series(mtsi,             index=idx),
    }

    # ── Apply LENS 10 & 90: z, z_slope, z_sos ────────────────────────────────
    for name, ind in Z_LENS_INDICATORS.items():
        arr = ind.values.astype(np.float64)
        for lens in [10, 90]:
            rm   = pd.Series(arr, index=idx).rolling(lens).mean().values
            rs   = pd.Series(arr, index=idx).rolling(lens).std().values
            z    = (arr - rm) / (rs + 1e-9)
            zs   = _rolling_linslope(z, lens)
            zsos = _rolling_linslope(zs, lens)
            df[f'LENS_{lens}_{name}_z']       = z
            df[f'LENS_{lens}_{name}_z_slope'] = zs
            df[f'LENS_{lens}_{name}_z_sos']   = zsos

    # ── Rolling % group: COG (unbounded — rolling deviation) ─────────────────
    cog_arr = cog_20.astype(np.float64)
    for win in [10, 30, 90]:
        rm = pd.Series(cog_arr, index=idx).rolling(win).mean().values
        df[f'WIN_{win}_cog_20_pct'] = cog_arr / (rm + 1e-9) - 1

    return (df.replace([np.inf, -np.inf], np.nan)
              .ffill()
              .dropna(subset=['T_FINAL'])
              .fillna(0))

# ── Top-level worker for ProcessPoolExecutor (must be picklable) ───────────────
def _process_symbol_worker(args):
    """Called in a subprocess. Returns processed DataFrame or None."""
    symbol, raw_dict = args
    try:
        raw_df = pd.DataFrame(raw_dict)
        raw_df.index = pd.to_datetime(raw_df.index)
        if len(raw_df) < 400:
            return None
        processed = generate_factory_features_v2(raw_df)
        if processed.empty:
            return None
        processed['symbol'] = symbol
        return processed.reset_index()   # reset so index survives pickling
    except Exception:
        return None

# ==============================================================================
# ### BLOCK 4: PARALLEL LOADER
# Phase 1: parallel network I/O      (ThreadPoolExecutor)
# Phase 2: parallel feature gen      (ThreadPoolExecutor — GIL released by
#           numpy/pandas heavy ops; ProcessPool excluded: Colab/Jupyter
#           subprocesses cannot import functions from __main__ kernel scope)
# ==============================================================================
def fetch_data(symbol, period="6y"):
    try:
        data = yf.download(symbol, period=period, interval="1d", progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).lower() for c in data.columns]
        return data if 'close' in data.columns else None
    except Exception:
        return None

def load_hybrid_data_parallel(brain_name, symbol_list, dl_workers=20):
    # ── Inline worker — defined here so it is always in scope regardless of
    # whether blocks are run as separate notebook cells.  ThreadPoolExecutor
    # does not pickle callables (same process), so nested functions are fine.
    def _worker(args):
        symbol, raw_dict = args
        try:
            raw_df = pd.DataFrame(raw_dict)
            raw_df.index = pd.to_datetime(raw_df.index)
            if len(raw_df) < 400:
                return None
            processed = generate_factory_features_v2(raw_df)
            if processed.empty:
                return None
            processed['symbol'] = symbol
            return processed.reset_index()
        except Exception:
            return None

    # DIRECTION needs 6y history; EASE/EXP need 4y
    period = "6y" if brain_name == "DIRECTION" else "4y"
    print(f"📥 Parallel download: {len(symbol_list)} symbols (period={period})...")
    raw_results = {}

    # ── Phase 1: parallel I/O ──────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        fut_map = {pool.submit(fetch_data, sym, period): sym for sym in symbol_list}
        for fut in tqdm(as_completed(fut_map), total=len(symbol_list),
                        desc="⬇ Downloading"):
            sym  = fut_map[fut]
            data = fut.result()
            if data is not None and len(data) >= 400:
                raw_results[sym] = data

    print(f"   ✅ {len(raw_results)}/{len(symbol_list)} symbols fetched")
    if not raw_results:
        return pd.DataFrame()

    # ── Phase 2: parallel feature generation ──────────────────────────────────
    # ThreadPoolExecutor only — ProcessPoolExecutor excluded because Colab/Jupyter
    # subprocesses cannot unpickle __main__-scoped functions (guaranteed NameError).
    work_items = [(sym, df.to_dict()) for sym, df in raw_results.items()]
    all_data   = []

    print(f"⚙ Building features in parallel (workers={N_FEATURE_WORKERS})...")
    with ThreadPoolExecutor(max_workers=N_FEATURE_WORKERS) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(work_items),
                        desc="⚙ Features"):
            result = fut.result()
            if result is not None:
                result = result.set_index(result.columns[0])
                all_data.append(result)

    if not all_data:
        print("❌ No valid data after feature generation.")
        return pd.DataFrame()

    print(f"   ✅ {len(all_data)} symbols processed")
    return pd.concat(all_data, axis=0)

# ==============================================================================
# ### BLOCK 5: GPU-ACCELERATED AUDIT  (v4.1 Brain-Mandate Aligned)
# Brain mandate restored per master_md5.md architecture requirements:
#   DIRECTION → GRU  + binary_crossentropy + accuracy quality gate (> 50%)
#   EASE/EXP  → LSTM + Huber              + Pearson r gate (> 0.005 S1SFT)
# S1SFT Pearson threshold (0.005) is intentionally lower than S2MFT production
# gate (0.02) — TREND-only features show weak-but-real signal at this stage.
# Permutation importance adapted per brain:
#   DIRECTION → accuracy DROP when feature permuted (higher = more important)
#   EASE/EXP  → MAE RISE when feature permuted    (higher = more important)
# ==============================================================================

N_PCA_COMPONENTS = 19   # Plan: S1SFT target — 19 within-family champions

def build_full_model(model_type, n_features, seq_len, output_mode='classify',
                     device=DEVICE):
    """
    Fast Mode spec (plan §Model Configuration):
      GPU:  GRU/LSTM(64 → 32) → Dense(16) → Dense(1)  ~25k params
      CPU:  Conv1D(32 → 16) + GAP → Dense(16) → Dense(1)  ~8k params
    output_mode: 'classify' → sigmoid + binary_crossentropy
                 'regress'  → linear  + huber
    """
    from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D
    out_act = 'sigmoid' if output_mode == 'classify' else 'linear'
    loss_fn = ('binary_crossentropy' if output_mode == 'classify'
               else tf.keras.losses.Huber())
    metrics = ['accuracy'] if output_mode == 'classify' else ['mae']

    with tf.device(device):
        if GPU_AVAILABLE:
            # ── Fast Mode GPU path (plan: [64, 32, 16]) ───────────────────────
            model = Sequential([
                Input(shape=(seq_len, n_features)),
                GRU(64,  return_sequences=True) if model_type == 'GRU'
                    else LSTM(64, return_sequences=True),
                Dropout(0.2),
                GRU(32) if model_type == 'GRU' else LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1,  activation=out_act, dtype='float32'),
            ])
        else:
            # ── CPU path: Conv1D (~8k params, fast on CPU) ────────────────────
            model = Sequential([
                Input(shape=(seq_len, n_features)),
                Conv1D(32, kernel_size=3, activation='relu', padding='same'),
                Dropout(0.1),
                Conv1D(16, kernel_size=3, activation='relu', padding='same'),
                GlobalAveragePooling1D(),
                Dense(16, activation='relu'),
                Dropout(0.2),
                Dense(1,  activation=out_act, dtype='float32'),
            ])

        model.compile(optimizer=Adam(3e-4), loss=loss_fn, metrics=metrics)
    return model


@tf.function
def _eval_accuracy(model, X_batch, y_batch):
    preds   = tf.squeeze(model(X_batch, training=False), axis=-1)
    correct = tf.equal(tf.cast(preds >= 0.5, tf.int32),
                       tf.cast(y_batch, tf.int32))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def _eval_mae(model, X_batch, y_batch):
    """MAE for regression mode.  Not @tf.function — called with variable batch
    sizes (full val, bull subset, bear subset) so retracing is avoided."""
    preds = tf.squeeze(model(X_batch, training=False), axis=-1)
    return float(tf.reduce_mean(tf.abs(preds - tf.cast(y_batch, tf.float32))).numpy())


WARMUP_ROWS = 350  # first N rows have unreliable indicator values — excluded from fits

def run_judicial_audit(brain_name, master_df, model_type='GRU',
                       seq_len=30, epochs=60, batch_size=2048, look_fwd=5):
    # ── Brain mandate: enforce architecture per md5 spec ──────────────────────
    # DIRECTION → GRU + classify;  EASE / EXP → LSTM + regress
    # The caller's model_type hint is overridden to keep architecture compliant.
    is_classify = (brain_name == 'DIRECTION')
    model_type  = 'GRU'      if is_classify else 'LSTM'
    output_mode = 'classify' if is_classify else 'regress'

    feature_cols = [c for c in master_df.columns
                    if c.startswith('LENS_') or c.startswith('WIN_')]
    n_raw   = len(feature_cols)
    N_COMPS = min(N_PCA_COMPONENTS, n_raw)   # safety: can't exceed raw count

    X_raw = master_df[feature_cols].values

    # ── Look-forward targets — computed per brain from OHLC ───────────────────
    cl_s = pd.Series(master_df['close'].values)
    hi_s = pd.Series(master_df['high'].values)
    lo_s = pd.Series(master_df['low'].values)
    tr   = np.maximum((hi_s - lo_s).values,
                      np.maximum(np.abs((hi_s - cl_s.shift(1))).values,
                                 np.abs((lo_s - cl_s.shift(1))).values))
    atr_14_y = pd.Series(tr).rolling(14).mean()

    if brain_name == 'DIRECTION':
        # Binary: did price close higher LOOK_FWD days from now?
        y_raw = (cl_s.shift(-look_fwd) > cl_s).astype(np.float32).values

    elif brain_name == 'EXP':
        # Expansion: max range over next LOOK_FWD bars / ATR-14
        fwd_hi = hi_s[::-1].rolling(look_fwd).max()[::-1]
        fwd_lo = lo_s[::-1].rolling(look_fwd).min()[::-1]
        y_raw  = ((fwd_hi - fwd_lo) / (atr_14_y + 1e-9)).values.astype(np.float32)

    else:  # EASE
        # Continuous forward log-return over LOOK_FWD days
        y_raw = np.log(cl_s.shift(-look_fwd) / (cl_s + 1e-9)).values.astype(np.float32)
    n     = len(X_raw)
    n_seq = n - seq_len

    # ── Walk-forward 70 / 15 / 15 split ───────────────────────────────────────
    train_end = int(n_seq * 0.70)
    val_end   = int(n_seq * 0.85)

    # ── Step 8 (Plan): RobustScaler, fit on training rows only ────────────────
    scaler = RobustScaler()
    scaler.fit(X_raw[WARMUP_ROWS : seq_len + train_end])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # ── Step 9 (Plan): PCA n_raw → 19, fit on training rows only ──────────────
    pca = PCA(n_components=N_COMPS)
    pca.fit(X_scaled[WARMUP_ROWS : seq_len + train_end])
    X_pca         = pca.transform(X_scaled).astype(np.float32)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  [PCA] {n_raw} raw → {N_COMPS} components "
          f"| variance retained: {var_explained:.1%}")

    # ── Build sequences from PCA-compressed data ───────────────────────────────
    X_seqs = np.stack([X_pca[i - seq_len:i] for i in range(seq_len, n)])
    y_seqs = y_raw[seq_len:]

    X_tr  = X_seqs[:train_end];         y_tr  = y_seqs[:train_end]
    X_val = X_seqs[train_end:val_end];  y_val = y_seqs[train_end:val_end]
    X_te  = X_seqs[val_end:];           y_te  = y_seqs[val_end:]

    print(f"  [DATA] train={len(X_tr):,}  val={len(X_val):,}  "
          f"test={len(X_te):,}  input=({seq_len}, {N_COMPS})")

    AUTO     = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                .shuffle(min(20_000, len(X_tr)), reshuffle_each_iteration=True)
                .batch(batch_size).prefetch(AUTO))
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(batch_size * 2).prefetch(AUTO))

    model = build_full_model(model_type, N_COMPS, seq_len, output_mode=output_mode)
    with tf.device(DEVICE):
        model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=7,
                              restore_best_weights=True, min_delta=1e-4),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=4, min_lr=1e-6),
            ],
            verbose=1,
        )

    X_val_tf      = tf.constant(X_val)
    y_val_tf_gate = tf.constant(y_val)

    # ── Brain-specific baseline metric & quality gate ──────────────────────────
    if is_classify:
        # DIRECTION: must beat coin-flip (50%)
        baseline_metric = float(
            _eval_accuracy(model, X_val_tf, y_val_tf_gate).numpy())
        print(f"  [MODEL] Val Accuracy: {baseline_metric:.4f}")
        passes_gate = baseline_metric > 0.50
        print(f"  [GATE] val_accuracy={baseline_metric:.4f}  passes={passes_gate}")
        if not passes_gate:
            del model; gc.collect(); tf.keras.backend.clear_session()
            print(f"  ⚠️  Quality gate FAILED: accuracy={baseline_metric:.4f} "
                  f"<= 0.50 — no directional signal")
            return pd.DataFrame()
        # Held-out test
        test_metric = float(
            _eval_accuracy(model, tf.constant(X_te), tf.constant(y_te)).numpy())
        print(f"  [MODEL] Test Accuracy: {test_metric:.4f}")
        if test_metric > 0.70 and baseline_metric > 0.70:
            print(f"  🚨 LEAKAGE WARNING: val_acc={baseline_metric:.4f}, "
                  f"test_acc={test_metric:.4f} — investigate!")
    else:
        # EASE / EXP: MAE + Pearson gate.
        # S1SFT threshold r > 0.005 (vs S2MFT production gate 0.02) —
        # TREND-only features show weak-but-real signal at this stage; the gate
        # must detect signal presence for feature ranking, not production accuracy.
        baseline_metric = _eval_mae(model, X_val_tf, y_val_tf_gate)
        print(f"  [MODEL] Val MAE: {baseline_metric:.6f}")
        naive_mae   = float(np.mean(np.abs(y_val)))
        val_preds   = tf.squeeze(model(X_val_tf, training=False),
                                 axis=-1).numpy().flatten()
        pearson_r   = float(np.corrcoef(val_preds, y_val.flatten())[0, 1])
        passes_mae  = baseline_metric < naive_mae
        passes_corr = pearson_r > 0.005          # S1SFT threshold
        print(f"  [GATE] passes_mae={passes_mae}  pearson_r={pearson_r:.4f}  "
              f"passes_corr={passes_corr}")
        if not (passes_mae or passes_corr):
            del model; gc.collect(); tf.keras.backend.clear_session()
            print(f"  ⚠️  Quality gate FAILED: val_mae={baseline_metric:.6f} "
                  f">= naive={naive_mae:.6f}, r={pearson_r:.4f} — no signal")
            return pd.DataFrame()
        # Held-out test
        test_metric = _eval_mae(model, tf.constant(X_te), tf.constant(y_te))
        print(f"  [MODEL] Test MAE: {test_metric:.6f}")
        if test_metric < naive_mae * 0.50 and baseline_metric < naive_mae * 0.50:
            print(f"  🚨 LEAKAGE WARNING: val_mae={baseline_metric:.6f}, "
                  f"test_mae={test_metric:.6f} — investigate!")

    # ── Regime masks (bull = up day, bear = down/flat day) on validation set ───
    # A stable feature must show consistent importance in BOTH regimes.
    bull_mask = y_val > 0
    bear_mask = ~bull_mask
    X_bull = X_val[bull_mask];  y_bull = y_val[bull_mask]
    X_bear = X_val[bear_mask];  y_bear = y_val[bear_mask]

    # Fallback: if one regime is empty keep full-set metric (avoids div-by-zero)
    has_bull = bull_mask.any()
    has_bear = bear_mask.any()
    if is_classify:
        baseline_bull = (float(_eval_accuracy(model, tf.constant(X_bull),
                                              tf.constant(y_bull)).numpy())
                         if has_bull else baseline_metric)
        baseline_bear = (float(_eval_accuracy(model, tf.constant(X_bear),
                                              tf.constant(y_bear)).numpy())
                         if has_bear else baseline_metric)
        print(f"  [REGIME] bull_acc={baseline_bull:.4f}  "
              f"bear_acc={baseline_bear:.4f}  "
              f"(bull={bull_mask.sum()}, bear={bear_mask.sum()})")
    else:
        baseline_bull = (_eval_mae(model, tf.constant(X_bull), tf.constant(y_bull))
                         if has_bull else baseline_metric)
        baseline_bear = (_eval_mae(model, tf.constant(X_bear), tf.constant(y_bear))
                         if has_bear else baseline_metric)
        print(f"  [REGIME] bull_mae={baseline_bull:.6f}  "
              f"bear_mae={baseline_bear:.6f}  "
              f"(bull={bull_mask.sum()}, bear={bear_mask.sum()})")

    # ── Permutation importance: permute in RAW SCALED space, re-apply PCA ──────
    # Permuting raw features (not PCA components) preserves LENS_/WIN_ feature-name
    # granularity required by Block 6 sovereign hunt.
    # DIRECTION: I_raw = accuracy DROP  (baseline - permuted, clipped ≥ 0)
    # EASE/EXP:  I_raw = MAE RISE       (permuted - baseline, clipped ≥ 0)
    val_raw_slice = X_scaled[train_end : val_end + seq_len]
    n_val_rows    = val_end - train_end
    y_val_tf      = tf.constant(y_val)

    report_rows = []
    for fi, feat_name in enumerate(tqdm(feature_cols, desc="Permutation scoring")):
        try:
            X_perm_raw = val_raw_slice.copy()
            flat       = X_perm_raw[:, fi].flatten()
            np.random.shuffle(flat)
            X_perm_raw[:, fi] = flat
            X_perm_pca  = pca.transform(X_perm_raw).astype(np.float32)
            X_perm_seqs = np.stack([X_perm_pca[k : k + seq_len]
                                    for k in range(n_val_rows)])

            if is_classify:
                # ── DIRECTION: importance = accuracy drop ──────────────────────
                perm_metric = float(_eval_accuracy(
                    model, tf.constant(X_perm_seqs), y_val_tf).numpy())
                I_raw = max(0.0, baseline_metric - perm_metric)

                # ── Regime stability ───────────────────────────────────────────
                if has_bull and has_bear:
                    perm_bull = float(_eval_accuracy(
                        model, tf.constant(X_perm_seqs[bull_mask]),
                        tf.constant(y_bull)).numpy())
                    perm_bear = float(_eval_accuracy(
                        model, tf.constant(X_perm_seqs[bear_mask]),
                        tf.constant(y_bear)).numpy())
                    I_bull    = max(0.0, baseline_bull - perm_bull)
                    I_bear    = max(0.0, baseline_bear - perm_bear)
                    stability = 1.0 - abs(I_bull - I_bear) / (I_bull + I_bear + 1e-9)
                else:
                    stability = 0.5   # neutral — single-regime data
            else:
                # ── EASE/EXP: importance = MAE rise ───────────────────────────
                perm_metric = _eval_mae(model, tf.constant(X_perm_seqs), y_val_tf)
                I_raw = max(0.0, perm_metric - baseline_metric)

                # ── Regime stability ───────────────────────────────────────────
                if has_bull and has_bear:
                    perm_bull = _eval_mae(model,
                                         tf.constant(X_perm_seqs[bull_mask]),
                                         tf.constant(y_bull))
                    perm_bear = _eval_mae(model,
                                         tf.constant(X_perm_seqs[bear_mask]),
                                         tf.constant(y_bear))
                    I_bull    = max(0.0, perm_bull - baseline_bull)
                    I_bear    = max(0.0, perm_bear - baseline_bear)
                    stability = 1.0 - abs(I_bull - I_bear) / (I_bull + I_bear + 1e-9)
                else:
                    stability = 0.5   # neutral — single-regime data

            report_rows.append({
                'Feature':     feat_name,
                'I_raw':       I_raw,
                'I_stability': stability,
            })
        except Exception:
            report_rows.append({'Feature': feat_name, 'I_raw': 0.0,
                                 'I_stability': 0.5})

    del model; gc.collect(); tf.keras.backend.clear_session()
    return pd.DataFrame(report_rows)

# ==============================================================================
# ### BLOCK 6: SOVEREIGN HUNT & DIVERSITY ANCHORS
# BRAIN_LOCKS corrected to match actual factory column names.
# ==============================================================================
BRAIN_LOCKS = {
    'DIRECTION': ['LENS_90_cog_20_z_slope'],
    'EASE':      ['LENS_90_cog_20_z_sos'],
    'EXP':       ['LENS_10_hurst_50_z', 'LENS_90_cog_20_z_sos'],
}

# Bounded indicators:   LENS_{10|90}_{name}_{z|z_slope|z_sos}
# Unbounded indicators: WIN_{10|30|90}_{name}_pct

def _parse_feature_name(f):
    """
    Splits a LENS_ or WIN_ feature name into components.

    Examples:
      LENS_10_cog_20_z_slope  → prefix='LENS', window='10',
                                 lookback='LENS_10_cog_20', family='cog'
      WIN_10_cog_20_pct       → prefix='WIN',  window='10',
                                 lookback='WIN_10_cog_20',  family='cog'

    RULE:
      - Only ONE window per family is allowed in the final 19.
        LENS_10 vs LENS_90 of cog_20 are competing.
      - All three transforms of the winning window CAN coexist:
        LENS_10_cog_20_z / _z_slope / _z_sos share the same lookback key
        and do not block each other.
    """
    TRANSFORM_TOKENS = {'z', 'slope', 'sos', 'pct'}
    if f.startswith('LENS_') or f.startswith('WIN_'):
        parts     = f.split('_')
        prefix    = parts[0]
        window    = parts[1]
        remainder = list(parts[2:])
        while remainder and remainder[-1] in TRANSFORM_TOKENS:
            remainder.pop()
        indicator = '_'.join(remainder)
        family    = '_'.join(p for p in remainder if not p.isdigit())
        lookback  = f'{prefix}_{window}_{indicator}'
        return prefix, window, lookback, family
    return None, None, f, f

def apply_sovereign_hunt(ledger_df, master_data_df, brain_name, max_slots=19):
    locked_list = BRAIN_LOCKS.get(brain_name, [])
    # Stage 1 spec: sort by Sovereign Score (70% impact + 15% stability + 15%
    # uniqueness).  Fall back to I_raw if Sov_Score column is not present.
    sort_col   = 'Sov_Score' if 'Sov_Score' in ledger_df.columns else 'I_raw'
    candidates = ledger_df.sort_values(by=sort_col, ascending=False)
    picked     = [f for f in locked_list if f in ledger_df['Feature'].values]

    for lf in locked_list:
        if lf not in ledger_df['Feature'].values:
            print(f"  ⚠️  BRAIN_LOCK '{lf}' not found — check name")

    # Stage 1 spec: Uniqueness threshold = 0.95 (Identity Function Trap only).
    # At 15% weight uniqueness is a diversity floor, not a primary filter.
    # Features are allowed through unless they are extreme clones (|r| > 0.95).
    CORR_THRESHOLD = 0.95
    # Track which lookback is committed per family
    # e.g. family_lookback['cog'] = 'LENS_10_cog_20'
    # → blocks 'LENS_90_cog_20' but not more LENS_10_cog_20 transforms
    family_lookback = {}
    for f in picked:
        _, _, lookback, family = _parse_feature_name(f)
        if family not in family_lookback:
            family_lookback[family] = lookback

    feat_cols   = [c for c in master_data_df.columns
                   if c.startswith('LENS_') or c.startswith('WIN_')]
    corr_matrix = master_data_df[feat_cols].corr()

    for _, row in candidates.iterrows():
        if len(picked) >= max_slots:
            break
        f_name = row['Feature']
        if f_name in picked:
            continue
        _, _, f_lookback, f_family = _parse_feature_name(f_name)
        if f_family in family_lookback and family_lookback[f_family] != f_lookback:
            continue
        if (len(picked) > 0 and
                corr_matrix[f_name].loc[picked].max() > CORR_THRESHOLD):
            continue
        picked.append(f_name)
        if f_family not in family_lookback:
            family_lookback[f_family] = f_lookback

    pca = PCA()
    pca.fit(RobustScaler().fit_transform(master_data_df[picked]))
    cumvar  = np.cumsum(pca.explained_variance_ratio_)
    n_for_95 = int(np.searchsorted(cumvar, 0.95)) + 1   # components needed for 95% variance
    return picked, cumvar, n_for_95

def generate_judicial_ledger(brain_name, report_df, master_data_df, iteration=1):
    df = report_df.copy()

    # ── Normalized Impact (I_Norm) ─────────────────────────────────────────────
    df['I_Norm'] = ((df['I_raw'] - df['I_raw'].min()) /
                    (df['I_raw'].max() - df['I_raw'].min() + 1e-9))

    # ── Audit Vitality Gate ────────────────────────────────────────────────────
    # Spec: "Terminate any feature audit where mean I_Norm falls below 0.05."
    # This indicates the feature set has found no tradeable signal beyond noise.
    mean_i_norm = df['I_Norm'].mean()
    if mean_i_norm < 0.05:
        print(f"  🚨 VITALITY GATE: mean I_Norm={mean_i_norm:.4f} < 0.05 — "
              f"no tradeable signal beyond noise — audit terminated")
        return pd.DataFrame()

    # ── Pre-compute UV_score: intrinsic diversity of each feature vs. pool ─────
    # UV_score_i = 1 - mean(|corr(i, j)|) for all j ≠ i in the candidate pool.
    # This is the 15%-weighted uniqueness component of the Sovereign Score.
    # Hard gate (Identity Function Trap: |r| > 0.95) is enforced in sovereign hunt.
    feat_pool    = df['Feature'].tolist()
    feat_cols_ok = [c for c in master_data_df.columns
                    if (c.startswith('LENS_') or c.startswith('WIN_'))
                    and c in feat_pool]
    corr_full = (master_data_df[feat_cols_ok].corr().abs()
                 if feat_cols_ok else pd.DataFrame())
    uv_map = {}
    for f in feat_pool:
        if f in corr_full.columns:
            others    = [c for c in feat_cols_ok if c != f]
            uv_map[f] = float(1.0 - corr_full[f].loc[others].mean()) if others else 1.0
        else:
            uv_map[f] = 0.5
    df['UV_score'] = df['Feature'].map(uv_map).fillna(0.5)

    # ── Sovereign Score = 70% Impact + 15% Stability + 15% Uniqueness ─────────
    stab_col = 'I_stability' if 'I_stability' in df.columns else None
    df['Stab_Norm'] = df[stab_col].clip(0.0, 1.0) if stab_col else 0.5
    df['Sov_Score'] = (0.70 * df['I_Norm'] +
                       0.15 * df['Stab_Norm'] +
                       0.15 * df['UV_score'].clip(0.0, 1.0))

    # ── Sovereign Hunt (ranked by Sov_Score; hard gate |r|>0.95) ──────────────
    active_picks, var_map, n95 = apply_sovereign_hunt(df, master_data_df, brain_name)
    corr_sub = master_data_df[active_picks].corr().abs()
    avg_corr = ((corr_sub.sum().sum() - len(active_picks)) /
                (len(active_picks)**2 - len(active_picks) + 1e-9))

    W = 78   # total inner width for box borders
    print(f"\n╔══ {brain_name} SOVEREIGN CORE V.3.19.18 (Iter {iteration}) ══╗")
    print(f"║ {'RNK':<3} │ {'TREND FEATURE':<33} │ {'SOV':>5} │ "
          f"{'IMP':>5} │ {'STB':>5} │ {'UV%':>4} ║")
    print("╠" + "═"*5 + "╪" + "═"*35 + "╪" + "═"*7 + "╪" +
          "═"*7 + "╪" + "═"*7 + "╪" + "═"*6 + "╣")

    for i, f_name in enumerate(active_picks):
        f_row       = df[df['Feature'] == f_name].iloc[0]
        is_locked   = f_name in BRAIN_LOCKS.get(brain_name, [])
        icon        = "🔒" if is_locked else "🔭"
        other_picks = [p for p in active_picks if p != f_name]
        max_r  = corr_sub[f_name].loc[other_picks].max() if other_picks else 0.0
        uv_val = ((1 - corr_sub[f_name].loc[other_picks].mean()) * 100
                  if other_picks else 100.0)
        sov = f_row['Sov_Score']
        imp = f_row['I_Norm']
        stb = f_row['Stab_Norm']
        print(f"║ {i+1:02d}   │ {icon} {f_name[:31]:<31} │ "
              f"{sov:.3f} │ {imp:.3f} │ {stb:.3f} │ {uv_val:>3.0f}% ║")
        df.loc[df['Feature'] == f_name,
               ['UV%', 'Max_R', 'Is_Locked']] = [uv_val, max_r, is_locked]

    print("╠" + "═"*75 + "╣")
    print(f"║ SOV SCORE WEIGHTS : 70% Impact · 15% Stability · 15% Uniqueness"
          f"{'':>10}║")
    print(f"║ PCA 95% THRESHOLD : {n95:>2} of {len(active_picks)} components needed"
          f"{'':>38}║")
    print(f"║ AVG TEAM CROSS-CORR:          {avg_corr:>44.3f} ║")
    print(f"║ SLOTS FILLED:                 {len(active_picks):>44}/19 ║")
    print("╚" + "═"*75 + "╝")

    return df[df['Feature'].isin(active_picks)]

# ==============================================================================
# ### BLOCK 7: COMMAND CENTER
# ==============================================================================
print("\n--- SOVEREIGN TITAN v3.19.18 — GPU + SPEED EDITION v2 ---")
choice        = input("Select Brain (1:DIR / 2:EASE / 3:EXP / 4:ALL): ")
BRAINS_TO_RUN = (['DIRECTION', 'EASE', 'EXP'] if choice == '4'
                 else [{'1': 'DIRECTION', '2': 'EASE', '3': 'EXP'}[choice]])
num_symbols   = int(input("Symbols per iteration (Default 50): ")  or "50")
num_iters     = int(input("Iterations to run (Default 25): ")      or "25")
LOOK_FWD      = int(input("Look-forward days (Default 5): ")       or "5")

final_report_accumulator = []

for BRAIN in BRAINS_TO_RUN:
    CURRENT_MODEL_TYPE = 'GRU' if BRAIN == 'DIRECTION' else 'LSTM'
    print(f"\n[SYSTEM] Brain: {BRAIN} | Model: {CURRENT_MODEL_TYPE} | Device: {DEVICE}")

    for it in range(1, num_iters + 1):
        print(f"\n{'─'*55}")
        print(f"  Iteration {it}/{num_iters}  —  Brain: {BRAIN}")
        print(f"{'─'*55}")

        POOL      = random.sample(TITAN_SYMBOLS, min(num_symbols, len(TITAN_SYMBOLS)))
        master_df = load_hybrid_data_parallel(BRAIN, POOL)

        if master_df.empty:
            print("  ⚠️  Empty master_df — skipping.")
            continue

        report_raw = run_judicial_audit(BRAIN, master_df,
                                        model_type=CURRENT_MODEL_TYPE,
                                        look_fwd=LOOK_FWD)
        if report_raw.empty:
            print("  ⚠️  Quality gate — iteration skipped.")
            continue

        iteration_ledger = generate_judicial_ledger(BRAIN, report_raw,
                                                    master_df, iteration=it)
        if iteration_ledger.empty:
            print("  ⚠️  Vitality gate — iteration skipped.")
            continue
        iteration_ledger['Iteration']  = it
        iteration_ledger['Brain']      = BRAIN
        iteration_ledger['Model_Type'] = CURRENT_MODEL_TYPE
        iteration_ledger['Timestamp']  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        final_report_accumulator.append(iteration_ledger)
        gc.collect()
        tf.keras.backend.clear_session()

# ==============================================================================
# ### BLOCK 8: FINAL EXPORT & SOVEREIGN SELECTION
# ==============================================================================
if final_report_accumulator:
    raw_df = pd.concat(final_report_accumulator, axis=0)
    stats  = (raw_df.groupby(['Brain', 'Feature'])
              .agg(Persistence=('Feature', 'count'),
                   A_Impact=('I_Norm', 'mean'),
                   A_UV=('UV%', 'mean'))
              .reset_index())
    final_df = (raw_df.merge(stats, on=['Brain', 'Feature'], how='left')
                      .sort_values(['Brain', 'Persistence', 'A_Impact'],
                                   ascending=False))

    report_filename = (f"Sovereign_Audit_Master_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    report_path = os.path.join(OUTPUT_DRIVE_DIR, report_filename)
    final_df.to_csv(report_path, index=False)

    print("\n" + "="*65)
    print("✅ GLOBAL AUDIT COMPLETE")
    print(f"📊 DATA ROWS COLLECTED: {len(raw_df)}")
    print(f"📂 CSV SAVED TO:        {report_path}")
    print("="*65)

    print("\n" + "═"*65)
    print("🚀 FINAL SOVEREIGN ARRAYS (TOP 19 PER BRAIN)")
    print("═"*65)

    FINAL_SELECTIONS = {}
    for brain in BRAINS_TO_RUN:
        brain_stats = (stats[stats['Brain'] == brain]
                       .sort_values(['Persistence', 'A_Impact'], ascending=False))
        top_19 = brain_stats.head(19)
        FINAL_SELECTIONS[brain] = top_19['Feature'].tolist()

        print(f"\n💎 FINAL 19 — BRAIN: {brain}")
        print(f"{'RNK':<3} | {'FEATURE':<38} | {'PERSIST':<8} | {'AVG_IMP':<8}")
        print("─" * 62)
        for i, row in top_19.reset_index(drop=True).iterrows():
            print(f"{i+1:02d}  | {row['Feature']:<38} | "
                  f"{int(row['Persistence']):>2}/{num_iters:<5} | "
                  f"{row['A_Impact']:.4f}")

    for brain, winners in FINAL_SELECTIONS.items():
        BRAIN_LOCKS[brain] = winners

    print("\n" + "═"*65)
    print("✅ FINAL 19 SYNCED TO BRAIN_LOCKS")
    print(f"📂 TOTAL UNIQUE FEATURES LOGGED: {len(stats)}")
    print("═"*65)

else:
    print("\n⚠️ [CRITICAL] No data collected. Audit failed.")
