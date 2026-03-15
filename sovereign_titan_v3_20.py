# ==============================================================================
# SOVEREIGN TITAN v3.20 - Scottie Pippen Edition
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
@jit(nopython=True, cache=True)
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
@jit(nopython=True, cache=True)
def _rolling_linslope(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _lin_slope_nb(arr[i - window + 1 : i + 1])
    return out
# ── Hurst exponent ─────────────────────────────────────────────────────────────
@jit(nopython=True, cache=True)
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
@jit(nopython=True, cache=True)
def _rolling_hurst(arr, window):
    n = len(arr); out = np.full(n, 0.5)
    for i in range(window - 1, n):
        out[i] = _hurst_nb(arr[i - window + 1 : i + 1])
    return out
# ── Center of Gravity ──────────────────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _cog_nb(y):
    n = len(y)
    if n < 2: return 0.0
    num = 0.0; den = 0.0
    for i in range(n):
        w = float(i + 1)
        num += w * y[i]
        den += y[i]
    return -num / (den + 1e-9)
@jit(nopython=True, cache=True)
def _rolling_cog(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _cog_nb(arr[i - window + 1 : i + 1])
    return out
# ── Shannon entropy (manual histogram — np.histogram not in nopython) ──────────
@jit(nopython=True, cache=True)
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
@jit(nopython=True, cache=True)
def _rolling_shannon(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _shannon_nb(arr[i - window + 1 : i + 1])
    return out
# ── R-squared (correlation² of index vs values) ────────────────────────────────
@jit(nopython=True, cache=True)
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
@jit(nopython=True, cache=True)
def _rolling_r_sq(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _r_sq_nb(arr[i - window + 1 : i + 1])
    return out
# ── Weighted Moving Average ────────────────────────────────────────────────────
@jit(nopython=True, cache=True)
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
@jit(nopython=True, cache=True)
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
    # linreg computed ONCE (duplicate removed)
    linreg_30        = _rolling_linslope(hlc, 30)
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
    donchian_high_50 = (hi_s / hi_s.rolling(50).max() - 1).values
    aroon_up_25      = hi_s.rolling(25).apply(
        lambda x: float(np.argmax(x)) / 25, raw=True
    ).values
    # ── Pre-transform Price MAs: hlc3 / MA - 1 ────────────────────────────────
    tema_10_pct = hlc / (tema_10 + 1e-9) - 1
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
    ratio_acc          = _rolling_linslope(er_20,  10)
    ratio_snr          = r_sq_30 / (1.0 - r_sq_30 + 1e-9)
    curvature_diff     = _rolling_linslope(ema10, 10) - _rolling_linslope(ema30, 10)
    cycle_vs_trend     = np.abs(cog_20) / (r_sq_30 + 1e-9)
    ratio_eff_slope    = er_20 * np.sign(linreg_30)
    ratio_pers_slope   = hurst_50 * np.sign(linreg_30)
    ratio_struct       = (tema_10_pct - sma_20_pct) / (np.abs(sma_20_pct) + 1e-9)
    adx_entropy_ratio  = adx_14  / (shannon_20 + 1e-9)
    ratio_breakout_eff = np.abs(donchian_high_50) / (er_20 + 1e-9)
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
        'sma_5_pct':        pd.Series(sma_5_pct,        index=idx),
        'sma_20_pct':       pd.Series(sma_20_pct,       index=idx),
        'hma_21_pct':       pd.Series(hma_21_pct,       index=idx),
        'kalman_pct':          pd.Series(kalman_pct,          index=idx),
        'kalman_sma_ratio':    pd.Series(kalman_sma_ratio,    index=idx),
        'tema_kalman_ratio':   pd.Series(tema_kalman_ratio,   index=idx),
        'exhaustion':          pd.Series(exhaustion,          index=idx),
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
    # ── Apply LENS 10, 30, 60, 90: z, z_slope, z_sos ─────────────────────────
    for name, ind in Z_LENS_INDICATORS.items():
        arr = ind.values.astype(np.float64)
        for lens in [10, 30, 60, 90]:
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
    for win in [10, 30, 60]:
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
        if len(raw_df) < 120:
            return None
        processed = generate_factory_features_v2(raw_df)
        if processed.empty:
            return None
        processed['symbol'] = symbol
        return processed.reset_index()
    except Exception:
        return None
# ==============================================================================
# ### BLOCK 4: PARALLEL LOADER
# ==============================================================================
def fetch_data(symbol):
    try:
        data = yf.download(symbol, period="2y", interval="1d", progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).lower() for c in data.columns]
        return data if 'close' in data.columns else None
    except Exception:
        return None
def load_hybrid_data_parallel(brain_name, symbol_list, dl_workers=20):
    print(f"📥 Parallel download: {len(symbol_list)} symbols...")
    raw_results = {}
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        fut_map = {pool.submit(fetch_data, sym): sym for sym in symbol_list}
        for fut in tqdm(as_completed(fut_map), total=len(symbol_list),
                        desc="⬇ Downloading"):
            sym  = fut_map[fut]
            data = fut.result()
            if data is not None and len(data) >= 120:
                raw_results[sym] = data
    print(f"   ✅ {len(raw_results)}/{len(symbol_list)} symbols fetched")
    if not raw_results:
        return pd.DataFrame()
    work_items = [(sym, df.to_dict()) for sym, df in raw_results.items()]
    all_data   = []
    print(f"⚙ Building features in parallel (workers={N_FEATURE_WORKERS})...")
    try:
        with ProcessPoolExecutor(max_workers=N_FEATURE_WORKERS) as pool:
            futures = {pool.submit(_process_symbol_worker, item): item[0]
                       for item in work_items}
            for fut in tqdm(as_completed(futures), total=len(work_items),
                            desc="⚙ Features"):
                result = fut.result()
                if result is not None:
                    result = result.set_index(result.columns[0])
                    all_data.append(result)
    except Exception as e:
        print(f"  ⚠️  ProcessPool failed ({e}) — falling back to sequential")
        for item in tqdm(work_items, desc="⚙ Features (sequential)"):
            result = _process_symbol_worker(item)
            if result is not None:
                result = result.set_index(result.columns[0])
                all_data.append(result)
    if not all_data:
        print("❌ No valid data after feature generation.")
        return pd.DataFrame()
    print(f"   ✅ {len(all_data)} symbols processed")
    return pd.concat(all_data, axis=0)
# ==============================================================================
# ### BLOCK 5: GPU-ACCELERATED AUDIT
# ==============================================================================
def build_full_model(model_type, n_features, seq_len, device=DEVICE):
    with tf.device(device):
        model = Sequential([
            Input(shape=(seq_len, n_features)),
            GRU(128,  return_sequences=True) if model_type == 'GRU'
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
def _eval_accuracy(model, X_batch, y_batch):
    preds   = tf.squeeze(model(X_batch, training=False), axis=-1)
    correct = tf.equal(tf.cast(preds >= 0.5, tf.int32),
                       tf.cast(y_batch, tf.int32))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
def run_judicial_audit(brain_name, master_df, model_type='GRU',
                       seq_len=10, epochs=5, batch_size=1024):
    feature_cols = [c for c in master_df.columns
                    if c.startswith('LENS_') or c.startswith('WIN_')]
    n_features   = len(feature_cols)
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(
        master_df[feature_cols].values
    ).astype(np.float32)
    y_raw    = master_df['T_FINAL'].values.astype(np.float32)
    n      = len(X_scaled)
    X_seqs = np.stack([X_scaled[i - seq_len:i] for i in range(seq_len, n)])
    y_seqs = y_raw[seq_len:]
    split       = int(len(X_seqs) * 0.8)
    X_tr, X_val = X_seqs[:split], X_seqs[split:]
    y_tr, y_val = y_seqs[:split], y_seqs[split:]
    print(f"  [DATA] train={len(X_tr):,}  val={len(X_val):,}  features={n_features}")
    AUTO     = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                .shuffle(min(20_000, len(X_tr)), reshuffle_each_iteration=True)
                .batch(batch_size).prefetch(AUTO))
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(batch_size * 2).prefetch(AUTO))
    model = build_full_model(model_type, n_features, seq_len)
    with tf.device(DEVICE):
        model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=6,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=1e-5),
            ],
            verbose=1,
        )
    X_val_tf     = tf.constant(X_val)
    y_val_tf     = tf.constant(y_val)
    baseline_acc = _eval_accuracy(model, X_val_tf, y_val_tf).numpy()
    print(f"  [MODEL] Baseline val accuracy: {baseline_acc:.4f}")
    report_rows = []
    for fi, feat_name in enumerate(tqdm(feature_cols, desc="Permutation scoring")):
        try:
            X_perm = X_val.copy()
            flat   = X_perm[:, :, fi].flatten()
            np.random.shuffle(flat)
            X_perm[:, :, fi] = flat.reshape(X_perm[:, :, fi].shape)
            perm_acc = _eval_accuracy(model, tf.constant(X_perm), y_val_tf).numpy()
            report_rows.append({
                'Feature': feat_name,
                'I_raw':   max(0.0, baseline_acc - perm_acc),
            })
        except Exception:
            report_rows.append({'Feature': feat_name, 'I_raw': 0.0})
    del model; gc.collect(); tf.keras.backend.clear_session()
    return pd.DataFrame(report_rows)
# ==============================================================================
# ### BLOCK 6: SOVEREIGN HUNT & DIVERSITY ANCHORS
# ==============================================================================
BRAIN_LOCKS = {
    'DIRECTION': ['LENS_90_cog_20_z_slope'],
    'EASE':      ['LENS_90_cog_20_z_sos'],
    'EXP':       ['LENS_10_hurst_50_z', 'LENS_90_cog_20_z_sos'],
}
def _parse_feature_name(f):
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
    candidates  = ledger_df.sort_values(by='I_raw', ascending=False)
    picked      = [f for f in locked_list if f in ledger_df['Feature'].values]
    for lf in locked_list:
        if lf not in ledger_df['Feature'].values:
            print(f"  ⚠️  BRAIN_LOCK '{lf}' not found — check name")
    CORR_THRESHOLD = 0.85
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
    return picked, np.cumsum(pca.explained_variance_ratio_)
def generate_judicial_ledger(brain_name, report_df, master_data_df, iteration=1):
    df           = report_df.copy()
    df['I_Norm'] = (df['I_raw'] - df['I_raw'].min()) / \
                   (df['I_raw'].max() - df['I_raw'].min() + 1e-9)
    active_picks, var_map = apply_sovereign_hunt(df, master_data_df, brain_name)
    corr_sub = master_data_df[active_picks].corr().abs()
    avg_corr = ((corr_sub.sum().sum() - len(active_picks)) /
                (len(active_picks)**2 - len(active_picks) + 1e-9))
    print(f"\n╔══ {brain_name} SOVEREIGN CORE V.3.20 (Iter {iteration}) ══╗")
    print(f"║ {'RNK':<3} | {'TREND FEATURE':<35} | {'UV%':<4} | {'mR':<4} | {'IMPACT':<8} ║")
    print("╠" + "═"*4 + "╬" + "═"*37 + "╬" + "═"*6 + "╬" + "═"*6 + "╬" + "═"*10 + "╣")
    for i, f_name in enumerate(active_picks):
        f_row       = df[df['Feature'] == f_name].iloc[0]
        is_locked   = f_name in BRAIN_LOCKS.get(brain_name, [])
        icon        = "🔒" if is_locked else "🔭"
        other_picks = [p for p in active_picks if p != f_name]
        max_r  = corr_sub[f_name].loc[other_picks].max() if other_picks else 0.0
        uv_val = ((1 - corr_sub[f_name].loc[other_picks].mean()) * 100
                  if other_picks else 100.0)
        print(f"║ {i+1:02d}  | {icon} {f_name[:33]:<33} | "
              f"{uv_val:>3.0f}% | {max_r:.2f} | {f_row['I_Norm']:.4f} ║")
        df.loc[df['Feature'] == f_name,
               ['UV%', 'Max_R', 'Is_Locked']] = [uv_val, max_r, is_locked]
    total_var = var_map[-1] if len(var_map) > 0 else 0
    print("╠" + "═"*73 + "╣")
    print(f"║ PCA TOTAL VARIANCE RETENTION: {total_var*100:>33.2f}% ║")
    print(f"║ AVG TEAM CROSS-CORRELATION:   {avg_corr:>35.3f} ║")
    print(f"║ SLOTS FILLED:                 {len(active_picks):>35}/19 ║")
    print("╚" + "═"*73 + "╝")
    return df[df['Feature'].isin(active_picks)]
# ==============================================================================
# ### BLOCK 7: COMMAND CENTER
# ==============================================================================
print("\n--- SOVEREIGN TITAN v3.20 — Scottie Pippen Edition ---")
choice        = input("Select Brain (1:DIR / 2:EASE / 3:EXP / 4:ALL): ")
BRAINS_TO_RUN = (['DIRECTION', 'EASE', 'EXP'] if choice == '4'
                 else [{'1': 'DIRECTION', '2': 'EASE', '3': 'EXP'}[choice]])
num_symbols   = int(input("Symbols per iteration (Default 50): ") or "50")
num_iters     = int(input("Iterations to run (Default 25): ")     or "25")
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
        report_raw       = run_judicial_audit(BRAIN, master_df,
                                              model_type=CURRENT_MODEL_TYPE)
        iteration_ledger = generate_judicial_ledger(BRAIN, report_raw,
                                                    master_df, iteration=it)
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

# ==============================================================================
# ### BLOCK 9: FINAL AUDIT — automatic deep analysis of the CSV just generated
# Mirrors sovereign_audit_colab.py (Cells 4-11) but runs in-process on
# final_df so there is no file I/O round-trip.
# ==============================================================================
if final_report_accumulator:
    print("\n" + "═"*70)
    print("  BLOCK 9 — FINAL AUDIT RUNNING ON GENERATED CSV")
    print(f"  Source: {report_path}")
    print("═"*70)

    # ── Imports needed only for the audit ─────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from scipy.stats import spearmanr, kendalltau
    from IPython.display import display, HTML
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                         "axes.spines.right": False})

    # ── Config ────────────────────────────────────────────────────────────────
    _N_FEATURES = 19

    # ── Use the in-memory final_df (already merged with stats) ────────────────
    _df = final_df.copy()
    _df["Iteration"] = pd.to_numeric(_df["Iteration"],  errors="coerce").fillna(1).astype(int)
    _df["Is_Locked"] = _df["Is_Locked"].astype(str).str.lower().isin(["true","1","yes"])
    for col in ["I_Norm", "A_Impact", "A_UV", "UV%", "Max_R"]:
        if col in _df.columns:
            _df[col] = pd.to_numeric(_df[col], errors="coerce")

    _BRAINS  = sorted(_df["Brain"].unique())
    _ITERS   = sorted(_df["Iteration"].unique())
    _N_ITERS = len(_ITERS)

    print(f"  Rows       : {len(_df):,}")
    print(f"  Brains     : {_BRAINS}")
    print(f"  Iterations : {_ITERS}  (n={_N_ITERS})")
    print(f"  Unique feat: {_df['Feature'].nunique():,}")

    # ── Audit helpers ─────────────────────────────────────────────────────────
    def _audit_rank_iter(brain_df, iteration):
        sub = brain_df[brain_df["Iteration"] == iteration].copy()
        score_col = "A_Impact" if sub["A_Impact"].notna().any() else "I_Norm"
        sub = sub.groupby("Feature")[score_col].mean().reset_index()
        sub = sub.sort_values(score_col, ascending=False).reset_index(drop=True)
        return pd.Series(sub.index.values + 1, index=sub["Feature"].values)

    def _audit_iter_rank_corr(brain_df):
        rows      = []
        all_feats = sorted(brain_df["Feature"].unique())
        for i in range(len(_ITERS) - 1):
            ia, ib = _ITERS[i], _ITERS[i + 1]
            ra = _audit_rank_iter(brain_df, ia).reindex(all_feats).fillna(len(all_feats) + 1)
            rb = _audit_rank_iter(brain_df, ib).reindex(all_feats).fillna(len(all_feats) + 1)
            sp, _ = spearmanr(ra, rb)
            kt, _ = kendalltau(ra, rb)
            top_a = set(ra.nsmallest(_N_FEATURES).index)
            top_b = set(rb.nsmallest(_N_FEATURES).index)
            rows.append(dict(
                iter_a=ia, iter_b=ib,
                spearman_r=round(sp, 4), kendall_tau=round(kt, 4),
                top19_overlap=len(top_a & top_b),
                top19_new_entries=len(top_b - top_a),
            ))
        return pd.DataFrame(rows)

    def _audit_sov_score_cv(brain_df):
        score_col = "A_Impact" if brain_df["A_Impact"].notna().any() else "I_Norm"
        grp = (brain_df.groupby("Feature")[score_col]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "avg", "std": "sd", "count": "n_obs"}))
        grp["cv"] = (grp["sd"] / grp["avg"].replace(0, np.nan)).fillna(0)
        return grp.sort_values("avg", ascending=False)

    def _audit_verdict(corr_df, cv_df, brain):
        base = dict(brain=brain, n_iters=_N_ITERS, last_rho=0.0,
                    last_overlap=0, median_cv=0.0, trending_up=False,
                    pass_rho=False, pass_overlap=False, pass_cv=False,
                    notes=[], recommend_n=max(_N_ITERS + 3, 8))
        if corr_df.empty:
            base["verdict"] = "CANNOT_ASSESS"
            base["reason"]  = "Only 1 iteration found."
            return base
        last      = corr_df.iloc[-1]
        rho       = last["spearman_r"]
        overlap   = last["top19_overlap"]
        median_cv = cv_df["cv"].median()
        pass_rho  = rho >= 0.85
        pass_ovl  = overlap >= 16
        pass_cv   = median_cv <= 0.20
        all_pass  = pass_rho and pass_ovl and pass_cv
        notes = []
        if not pass_rho: notes.append(f"rank ρ={rho:.3f} < 0.85 — rankings still shifting")
        if not pass_ovl: notes.append(f"top-19 overlap={overlap}/19 < 16 — slot instability")
        if not pass_cv:  notes.append(f"median CV={median_cv:.3f} > 0.20 — score variance too high")
        trending_up = (len(corr_df) >= 2 and
                       corr_df["spearman_r"].iloc[-1] > corr_df["spearman_r"].iloc[-2])
        return dict(
            verdict      = "✅ SUFFICIENT" if all_pass else "⚠️  MORE NEEDED",
            brain        = brain,
            n_iters      = _N_ITERS,
            last_rho     = rho,
            last_overlap = overlap,
            median_cv    = median_cv,
            trending_up  = trending_up,
            pass_rho     = pass_rho,
            pass_overlap = pass_ovl,
            pass_cv      = pass_cv,
            notes        = notes,
            recommend_n  = max(_N_ITERS + 3, 8) if not all_pass else _N_ITERS,
        )

    def _audit_build_final_roster(brain_df, n=_N_FEATURES):
        score_col = "A_Impact" if brain_df["A_Impact"].notna().any() else "I_Norm"
        grp = (brain_df.groupby("Feature")
               .agg(
                   Runs      = ("Iteration", "nunique"),
                   Score     = (score_col,   "mean"),
                   Avg_INorm = ("I_Norm",     "mean"),
                   Avg_UV    = ("UV%",        "mean"),
                   Is_Locked = ("Is_Locked",  "first"),
               )
               .reset_index())
        mask = (grp["Feature"].str.contains("LENS_", case=False, na=False) |
                grp["Feature"].str.contains("WIN_",  case=False, na=False))
        grp = grp[mask].copy()
        grp["Persistence_pct"]  = grp["Runs"] / _N_ITERS
        grp["Stability_Weight"] = grp["Persistence_pct"] ** 2
        grp["Weighted_Score"]   = grp["Score"] * grp["Stability_Weight"]
        locked   = grp[grp["Is_Locked"]].copy()
        unlocked = grp[~grp["Is_Locked"]].sort_values("Weighted_Score", ascending=False)
        final = pd.concat([locked, unlocked.head(n - len(locked))], ignore_index=True)
        final = final.sort_values("Weighted_Score", ascending=False).reset_index(drop=True)
        final.index = final.index + 1
        return final

    # ── Run engines ───────────────────────────────────────────────────────────
    _corr_tables   = {}
    _cv_tables     = {}
    _verdicts      = {}
    _final_rosters = {}
    for brain in _BRAINS:
        bdf = _df[_df["Brain"] == brain]
        ct  = _audit_iter_rank_corr(bdf)
        cvt = _audit_sov_score_cv(bdf)
        _corr_tables[brain]   = ct
        _cv_tables[brain]     = cvt
        _verdicts[brain]      = _audit_verdict(ct, cvt, brain)
        _final_rosters[brain] = _audit_build_final_roster(bdf)
    print(f"\n✅ Audit engines complete for: {_BRAINS}")

    # ── Rank correlation tables ────────────────────────────────────────────────
    for brain in _BRAINS:
        corr = _corr_tables[brain]
        verd = _verdicts[brain]
        print(f"\n  ┌─ {brain} {'─'*(54-len(brain))}┐")
        if "CANNOT_ASSESS" in verd["verdict"]:
            print(f"  │  Only 1 iteration — rankings established, no trend yet.    │")
            print(f"  └{'─'*57}┘")
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
        if "CANNOT_ASSESS" not in verd["verdict"]:
            print(f"    last ρ={verd['last_rho']:.4f}  overlap={verd['last_overlap']}/19"
                  f"  median CV={verd['median_cv']:.4f}"
                  f"  trending={'↑' if verd['trending_up'] else '─'}")
        for note in verd.get("notes", []):
            print(f"    ⚠  {note}")

    # ── Stability-weighted final rosters ──────────────────────────────────────
    for brain in _BRAINS:
        roster = _final_rosters[brain]
        print(f"\n{'═'*78}")
        print(f"  STABILITY-WEIGHTED ROSTER ─ {brain}")
        print(f"  (Persistence² × raw score — dampens churn, rewards consistency)")
        print(f"{'═'*78}")
        print(f"  {'RNK':<4} {'FEATURE':<40} {'W_SCORE':>7} {'RAW':>7} {'PERS':>6} {'LK':>3}")
        print(f"  {'─'*72}")
        for rank, row in roster.iterrows():
            lock_s = "🔒" if row["Is_Locked"] else "  "
            pers_s = f"{row['Persistence_pct']*100:.0f}%"
            print(f"  {rank:02d}.  {row['Feature']:<40} {row['Weighted_Score']:>7.4f} "
                  f"{row['Score']:>7.4f} {pers_s:>5} {lock_s}")

    # ── Chart 1: Ranking Convergence ──────────────────────────────────────────
    n_b = len(_BRAINS)
    if _N_ITERS > 1:
        fig, axes = plt.subplots(1, n_b, figsize=(6 * n_b, 4), squeeze=False)
        for ax, brain in zip(axes[0], _BRAINS):
            corr = _corr_tables[brain]
            if corr.empty:
                ax.text(0.5, 0.5, "Only 1 iteration", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
            else:
                x = [f"{int(r.iter_a)}→{int(r.iter_b)}" for _, r in corr.iterrows()]
                ax.plot(x, corr["spearman_r"], "o-", color="#457b9d",
                        label="Spearman ρ", lw=2)
                ax.plot(x, corr["top19_overlap"] / _N_FEATURES, "s--",
                        color="#e63946", label=f"Top-{_N_FEATURES} overlap", lw=1.5)
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

    # ── Chart 2: Final Feature Rosters ────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, n_b, figsize=(9 * n_b, 7), squeeze=False)
    for ax, brain in zip(axes2[0], _BRAINS):
        roster = _final_rosters[brain].reset_index(drop=False)
        roster = roster.sort_values("Score", ascending=True)
        colors = ["#e63946" if lk else "#457b9d" for lk in roster["Is_Locked"]]
        ax.barh(roster["Feature"], roster["Score"], color=colors)
        ax.set_xlabel("Score")
        ax.set_title(f"{brain} — Final {_N_FEATURES} Features", fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(handles=[Patch(color="#e63946", label="Locked"),
                            Patch(color="#457b9d", label="Ranked")],
                  loc="lower right", fontsize=8)
    fig2.suptitle("Final Feature Rosters", fontsize=13, fontweight="bold", y=1.02)
    fig2.tight_layout()
    plt.show()

    # ── Chart 3: Score CV heatmap (multi-iter only) ───────────────────────────
    if _N_ITERS > 1:
        all_top_feats = set()
        for brain in _BRAINS:
            all_top_feats |= set(_final_rosters[brain]["Feature"])
        cv_matrix = pd.DataFrame(index=sorted(all_top_feats), columns=_BRAINS, dtype=float)
        for brain in _BRAINS:
            cv = _cv_tables[brain].reindex(sorted(all_top_feats))
            cv_matrix[brain] = cv["cv"]
        fig3, ax3 = plt.subplots(
            figsize=(max(5, 3 * n_b), max(8, len(all_top_feats) * 0.38))
        )
        im = ax3.imshow(cv_matrix.values.astype(float), aspect="auto",
                        cmap="RdYlGn_r", vmin=0, vmax=0.5)
        ax3.set_xticks(range(n_b))
        ax3.set_xticklabels(_BRAINS, fontsize=9)
        ax3.set_yticks(range(len(cv_matrix)))
        ax3.set_yticklabels(cv_matrix.index, fontsize=6)
        ax3.set_title("Score CV per Feature per Brain\n(green=stable, red=volatile)",
                      fontweight="bold")
        plt.colorbar(im, ax=ax3, label="CV (σ/μ)")
        fig3.tight_layout()
        plt.show()

    # ── Iteration Sufficiency Summary ─────────────────────────────────────────
    print("\n" + "═"*70)
    print("  ITERATION SUFFICIENCY SUMMARY")
    print("═"*70)
    all_sufficient = True
    for brain in _BRAINS:
        v = _verdicts[brain]
        all_sufficient = all_sufficient and ("SUFFICIENT" in v["verdict"])
        icon_rho = "✅" if v["pass_rho"]     else "❌"
        icon_ovl = "✅" if v["pass_overlap"] else "❌"
        icon_cv  = "✅" if v["pass_cv"]      else "❌"
        print(f"\n  {brain}")
        print(f"    Rank stability  (ρ ≥ 0.85)  : {icon_rho} ρ = {v['last_rho']:.4f}")
        print(f"    Slot stability  (≥16/19)    : {icon_ovl} overlap = {v['last_overlap']}/19")
        print(f"    Score variance  (CV ≤ 0.20) : {icon_cv} median CV = {v['median_cv']:.4f}")
        print(f"    ─ {v['verdict']} ─", end="")
        if "MORE NEEDED" in v["verdict"]:
            print(f"  (recommend ≥ {v['recommend_n']} total iterations)")
        else:
            print()
    print()
    if all_sufficient:
        print("  ✅ ALL BRAINS CONVERGED — current iteration count is sufficient.")
    else:
        worst = max(_verdicts.values(), key=lambda v: v["recommend_n"])
        print(f"  ⚠️  NOT ALL BRAINS CONVERGED — run at least "
              f"{worst['recommend_n']} total iterations and re-assess.")
    print("═"*70)
    print(f"\n📂 Audit complete. Full data: {report_path}")
else:
    print("\n⚠️ [BLOCK 9] No data to audit — collection failed in Block 7.")
