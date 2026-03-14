# =============================================================================
# SOVEREIGN TITAN v3.17.9 — CORRECTED EDITION
# Fixes vs original:
#   1. Single master CSV  (Sovereign_Audit_Master_YYYYMMDD_HHMMSS.csv)
#   2. EASE brain reads parquet with proper index alignment + column guard
#   3. Real permutation importance (was mean/std — meaningless)
#   4. Correct model input shape (seq_len, n_features) — was (20, 1)
#   5. re, gc imports added; export_titan_ledger moved before command center
#   6. LOOK_FWD input applies to DIRECTION shift and EXP range/avg_range
#   7. generate_expansion_target removed — EXP target is inline look-forward
# =============================================================================
import os, gc, re, numpy as np, pandas as pd, yfinance as yf, random
from scipy import signal as sp_signal
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm
import warnings
from google.colab import drive
warnings.filterwarnings('ignore')

# ── Drive & paths ─────────────────────────────────────────────────────────────
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive', force_remount=True)

DRIVE_DB_PATH    = '/content/drive/MyDrive/backtest_results/FRICTION_MASTER_DB.parquet'
TEST_NAME        = "Sovereign_Titan_v3.17.9_Ultimate"
OUTPUT_DRIVE_DIR = f'/content/drive/MyDrive/judicial_results/{TEST_NAME}/'
os.makedirs(OUTPUT_DRIVE_DIR, exist_ok=True)

# Single master CSV — path locked at run-start
_RUN_TS    = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_CSV = os.path.join(OUTPUT_DRIVE_DIR, f"Sovereign_Audit_Master_{_RUN_TS}.csv")

# ── GPU ───────────────────────────────────────────────────────────────────────
_gpus = tf.config.list_physical_devices('GPU')
for _g in _gpus:
    tf.config.experimental.set_memory_growth(_g, True)
DEVICE = '/device:GPU:0' if _gpus else '/cpu:0'
print(f"[SYSTEM] Compute : {DEVICE}")
print(f"[SYSTEM] Output  : {MASTER_CSV}")

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

# =============================================================================
# UTILITIES
# =============================================================================
def get_hurst_fast(series, window=100):
    def hurst_calc(ts):
        v = ts[np.isfinite(ts)]
        if len(v) < 25 or np.all(v == v[0]): return 0.5
        lags = range(2, min(20, len(v) // 2))
        try:
            tau = [np.std(np.subtract(v[lag:], v[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]
        except: return 0.5
    return series.rolling(window, min_periods=25).apply(hurst_calc, raw=True)

def fetch_yfinance_data(symbol, period='2y'):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty or len(df) < 200: return pd.DataFrame()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except: return pd.DataFrame()

def generate_direction_target(df, look_fwd=1):
    """Binary: 1 if close N bars forward > current close, else 0."""
    return (df['close'].shift(-look_fwd) > df['close']).astype(int)

# =============================================================================
# FEATURE FACTORY
# =============================================================================
def generate_heavy_physics(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    h, l, c, v = df['high'], df['low'], df['close'], df['volume']
    tp  = (h + l + c) / 3
    idx = df.index
    seeds = pd.DataFrame(index=idx)
    hlc   = tp.values.astype(np.float64)
    hi, lo, cl, vol = h.values, l.values, c.values, v.values

    def _rolling_wma(series, window):
        weights = np.arange(1, window + 1)
        return pd.Series(series).rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True).values

    def _kalman_numba(series):
        n = len(series); kf = np.zeros(n); kf[0] = series[0]
        q, r, p = 1e-5, 1e-2, 1.0
        for i in range(1, n):
            p += q; k = p / (p + r)
            kf[i] = kf[i-1] + k * (series[i] - kf[i-1])
            p *= (1 - k)
        return kf

    def _rolling_r_sq(series, window):
        def r_sq(x):
            if np.all(x == x[0]): return 0.0
            xa = np.arange(len(x), dtype=float)
            s, b = np.polyfit(xa, x, 1)
            ss_res = np.sum((x - (s * xa + b)) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-9)
        return pd.Series(series).rolling(window).apply(r_sq, raw=True).values

    def _rolling_hurst(series, window):
        return get_hurst_fast(pd.Series(series, index=idx),
                              window=window).fillna(0.5).values

    def _rolling_shannon(series, window):
        def shannon_ent(x):
            if np.all(x == x[0]): return 0.0
            hist, _ = np.histogram(x, bins=10, range=(x.min(), x.max()))
            ps = hist.astype(float) / (hist.sum() + 1e-9); ps = ps[ps > 0]
            return -np.sum(ps * np.log2(ps + 1e-10))
        return pd.Series(series).rolling(window).apply(shannon_ent, raw=True).values

    def _rolling_linslope(series, window):
        return pd.Series(series).rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0.0,
            raw=True).values

    # ── Moving Averages & Filters ─────────────────────────────────────────────
    ema30   = pd.Series(hlc).ewm(span=30).mean().values
    tema_30 = (3*ema30
               - 3*pd.Series(ema30).ewm(span=30).mean().values
               + pd.Series(pd.Series(ema30).ewm(span=30).mean()).ewm(span=30).mean().values)
    sma_20  = pd.Series(hlc).rolling(20).mean().values
    sma_30  = pd.Series(hlc).rolling(30).mean().values
    hma_21  = pd.Series(2 * _rolling_wma(hlc, 10) - _rolling_wma(hlc, 21)).rolling(5).mean().values
    kalman  = _kalman_numba(hlc)

    # ── Efficiency & R-Squared ────────────────────────────────────────────────
    er_10  = (pd.Series(hlc).diff(10).abs() /
              (pd.Series(hlc).diff().abs().rolling(10).sum() + 1e-9)).values
    er_20  = (pd.Series(hlc).diff(20).abs() /
              (pd.Series(hlc).diff().abs().rolling(20).sum() + 1e-9)).values
    vidya_cmo_20 = (pd.Series(hlc).diff().rolling(20).sum() /
                    (pd.Series(hlc).diff().abs().rolling(20).sum() + 1e-9)).values
    r_sq_20, r_sq_30 = [_rolling_r_sq(hlc, w) for w in [20, 30]]

    # ── Linear Regression & Slopes (raw — not divided by close) ─────────────
    slope_10  = _rolling_linslope(hlc, 10)
    slope_20  = _rolling_linslope(hlc, 20)
    slope_60  = _rolling_linslope(hlc, 60)
    linreg_30 = _rolling_linslope(hlc, 30)          # alias used in ratios + dict
    logistic_prob_30 = 1.0 / (1.0 + np.exp(-linreg_30 / (np.nanstd(linreg_30) + 1e-9)))

    # ── Volatility ────────────────────────────────────────────────────────────
    cl_prev = np.roll(cl, 1); cl_prev[0] = cl[0]
    tr      = np.maximum(hi - lo, np.maximum(np.abs(hi - cl_prev), np.abs(lo - cl_prev)))
    atr_14  = pd.Series(tr).rolling(14).mean().values

    # ── ADX Suite ─────────────────────────────────────────────────────────────
    hi_p, lo_p = np.roll(hi, 1), np.roll(lo, 1)
    plus_dm  = np.where((hi-hi_p) > (lo_p-lo), np.maximum(hi-hi_p, 0), 0)
    minus_dm = np.where((lo_p-lo) > (hi-hi_p), np.maximum(lo_p-lo, 0), 0)
    pdi14 = 100 * (pd.Series(plus_dm).rolling(14).mean()  / (atr_14 + 1e-9))
    mdi14 = 100 * (pd.Series(minus_dm).rolling(14).mean() / (atr_14 + 1e-9))
    adx_14 = (100 * np.abs(pdi14 - mdi14) / (pdi14 + mdi14 + 1e-9)).rolling(14).mean().values

    # ── Structural ────────────────────────────────────────────────────────────
    hi_s = pd.Series(hi); lo_s = pd.Series(lo)
    dispersion_30    = np.std(np.stack([hlc/sma_30-1, hlc/tema_30-1, hlc/kalman-1], axis=1), axis=1)
    donchian_high_20 = (hi / hi_s.rolling(20).max() - 1).values
    donchian_high_50 = (hi / hi_s.rolling(50).max() - 1).values
    quadratic_a_20   = pd.Series(hlc).rolling(20).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 2)[0] if len(x) > 2 else 0.0,
        raw=True).values

    # ── Aroon Up (25, 20, 60) ─────────────────────────────────────────────────
    aroon_up_25 = hi_s.rolling(25).apply(lambda x: float(np.argmax(x)) / 25, raw=True).values
    aroon_up20  = hi_s.rolling(20).apply(lambda x: float(np.argmax(x)) / 20, raw=True).values
    aroon_up60  = hi_s.rolling(60).apply(lambda x: float(np.argmax(x)) / 60, raw=True).values

    # ── Multi-period Shannon entropy ──────────────────────────────────────────
    shannon_10 = _rolling_shannon(hlc, 10)
    shannon_20 = _rolling_shannon(hlc, 20)
    shannon_40 = _rolling_shannon(hlc, 40)

    # ── Dominant cycle (spectral peak period, window=20) ─────────────────────
    def _dominant_cycle(x):
        x = x - x.mean()
        if np.all(x == 0): return np.nan
        f, pxx = sp_signal.periodogram(x)
        if len(f) < 2: return np.nan
        peak = pxx[1:].argmax() + 1
        return 1.0 / f[peak] if f[peak] > 0 else np.nan
    dominant_cycle_20 = pd.Series(hlc).rolling(20).apply(_dominant_cycle, raw=True).values

    # ── Price MA percentages ──────────────────────────────────────────────────
    tema_30_pct = hlc / (tema_30 + 1e-9) - 1
    sma_20_pct  = hlc / (sma_20  + 1e-9) - 1
    hma_21_pct  = hlc / (hma_21  + 1e-9) - 1
    kalman_pct  = hlc / (kalman  + 1e-9) - 1

    # ── MTSI — 2-bar VWAP deviation, smoothed ────────────────────────────────
    _cl_s  = pd.Series(cl,  index=idx)
    _tp_v  = pd.Series(tp * vol, index=idx)
    _vol_s = pd.Series(vol, index=idx)
    mtsi   = (_cl_s - (_tp_v.rolling(2).sum() / (_vol_s.rolling(2).sum() + 1e-9))
              ).ewm(span=3).mean().values

    # ── Hurst (50) ────────────────────────────────────────────────────────────
    hurst_50 = _rolling_hurst(hlc, 50)

    # ── Ratios — log-stabilised to compress outliers before z-scoring ─────────
    # _lrat(n, d)  = sign(n/d) * log(|n/d|)   robust to extreme denominators
    # _lprod(n, d) = sign(n)   * log(|n*d| + ε)  for products
    # _ldiff(x)    = sign(x)   * log(|x|   + ε)  for differences
    _lrat  = lambda n, d: np.sign(n / (d + 1e-9)) * (np.log(np.abs(n) + 1e-9) - np.log(np.abs(d) + 1e-9))
    _lprod = lambda n, d: np.sign(n) * np.log(np.abs(n * d) + 1e-9)
    _ldiff = lambda x:    np.sign(x) * np.log(np.abs(x)     + 1e-9)

    ratio_acc         = _lrat(slope_10,  slope_20)
    ratio_snr         = _lrat(slope_20,  atr_14)
    ratio_eff_slope   = _lprod(slope_20, er_20)
    ratio_pers_slope  = _lprod(slope_20, hurst_50)
    ratio_struct      = _lrat(r_sq_20,   shannon_20)
    kalman_sma_ratio  = _lrat(kalman,    sma_20)
    tema_kalman_ratio = _lrat(tema_30,   kalman)
    curvature_diff    = _ldiff(slope_10  - slope_60)
    cycle_vs_trend    = _lrat(dominant_cycle_20, linreg_30)
    ratio_breakout_eff = _lrat(donchian_high_20, er_20)
    adx_entropy_ratio = _lrat(adx_14,    shannon_20)
    exhaustion_60     = _lrat(cl,        hi_s.rolling(60).max().values)

    # ── Z-Lens Integration (40 features × 3 lengths × 3 transforms = 360 cols)
    Z_LENS_INDICATORS = {
        # Bounded oscillators / efficiency
        'er_20':             pd.Series(er_20,            index=idx),
        'vidya_cmo_20':      pd.Series(vidya_cmo_20,     index=idx),
        'r_sq_30':           pd.Series(r_sq_30,          index=idx),
        'hurst_50':          pd.Series(hurst_50,         index=idx),
        'shannon_20':        pd.Series(shannon_20,       index=idx),
        'adx_14':            pd.Series(adx_14,           index=idx),
        'logistic_prob_30':  pd.Series(logistic_prob_30, index=idx),
        'aroon_up_25':       pd.Series(aroon_up_25,      index=idx),
        'donchian_high_50':  pd.Series(donchian_high_50, index=idx),
        'dispersion_30':     pd.Series(dispersion_30,    index=idx),
        # Slopes (raw)
        'lr_slope_30':       pd.Series(linreg_30,        index=idx),
        # Price MA percentages
        'tema_30_pct':       pd.Series(tema_30_pct,      index=idx),
        'sma_20_pct':        pd.Series(sma_20_pct,       index=idx),
        'hma_21_pct':        pd.Series(hma_21_pct,       index=idx),
        'kalman_pct':        pd.Series(kalman_pct,       index=idx),
        # VWAP deviation
        'mtsi':              pd.Series(mtsi,             index=idx),
        # Multi-period slopes
        'lr_slope10':        pd.Series(slope_10,         index=idx),
        'lr_slope20':        pd.Series(slope_20,         index=idx),
        'lr_slope60':        pd.Series(slope_60,         index=idx),
        # Multi-period efficiency & R²
        'er10':              pd.Series(er_10,            index=idx),
        'r_sq20':            pd.Series(r_sq_20,          index=idx),
        # Multi-period entropy
        'shannon10':         pd.Series(shannon_10,       index=idx),
        'shannon40':         pd.Series(shannon_40,       index=idx),
        # Multi-period Aroon
        'aroon_up20':        pd.Series(aroon_up20,       index=idx),
        'aroon_up60':        pd.Series(aroon_up60,       index=idx),
        # Channel & curvature
        'donchian_high20':   pd.Series(donchian_high_20, index=idx),
        'quadratic_a20':     pd.Series(quadratic_a_20,   index=idx),
        'dominant_cycle20':  pd.Series(dominant_cycle_20,index=idx),
        # Ratios
        'ratio_acc':         pd.Series(ratio_acc,        index=idx),
        'ratio_snr':         pd.Series(ratio_snr,        index=idx),
        'ratio_eff_slope':   pd.Series(ratio_eff_slope,  index=idx),
        'ratio_pers_slope':  pd.Series(ratio_pers_slope, index=idx),
        'ratio_struct':      pd.Series(ratio_struct,     index=idx),
        'kalman_sma_ratio':  pd.Series(kalman_sma_ratio, index=idx),
        'tema_kalman_ratio': pd.Series(tema_kalman_ratio,index=idx),
        'curvature_diff':    pd.Series(curvature_diff,   index=idx),
        'cycle_vs_trend':    pd.Series(cycle_vs_trend,   index=idx),
        'ratio_breakout_eff':pd.Series(ratio_breakout_eff,index=idx),
        'adx_entropy_ratio': pd.Series(adx_entropy_ratio,index=idx),
        'exhaustion_60':     pd.Series(exhaustion_60,    index=idx),
    }
    for name, ind in Z_LENS_INDICATORS.items():
        arr = ind.values.astype(np.float64)
        s   = pd.Series(arr)
        for lens in [10, 30, 90]:
            z  = (arr - s.rolling(lens).mean().values) / (s.rolling(lens).std().values + 1e-9)
            zs = _rolling_linslope(z, lens)
            seeds[f'LENS_{lens}_{name}_z']       = z
            seeds[f'LENS_{lens}_{name}_z_slope'] = zs
            seeds[f'LENS_{lens}_{name}_z_sos']   = _rolling_linslope(zs, lens)

    return seeds.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# =============================================================================
# DATA LOADING
# DIRECTION : yfinance  → binary close[+LOOK_FWD] > close[0]
# EXP       : yfinance  → next LOOK_FWD bar range / 14-bar avg range
# EASE      : parquet   → ease_val (pre-scored, no shift needed)
# =============================================================================
def load_hybrid_data(brain_type, db_path, symbols, look_fwd=1):
    combined = []

    if brain_type == 'EASE':
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Parquet not found: {db_path}")
        master = pd.read_parquet(db_path)
        master.columns = [str(c).strip().lower() for c in master.columns]
        if 'ease_val' not in master.columns:
            raise ValueError(
                "Parquet has no 'ease_val' column.\n"
                f"  Found: {list(master.columns[:10])}...")
        if 'symbol' in master.columns:
            master = master[master['symbol'].isin(symbols)]

        for sym in tqdm(symbols, desc="Building EASE Tensor"):
            df_sym = (master[master['symbol'] == sym].copy()
                      if 'symbol' in master.columns else master.copy())
            if len(df_sym) < 200:
                continue
            df_sym = df_sym.reset_index(drop=True)
            try:
                phys    = generate_heavy_physics(df_sym)
                ease    = df_sym['ease_val'].reset_index(drop=True).rename('ease_val')
                aligned = phys.join(ease, how='inner').dropna()
                if len(aligned) > 50:
                    combined.append(aligned)
            except Exception as e:
                print(f"  ⚠ EASE {sym}: {e}")

    else:
        for sym in tqdm(symbols, desc=f"Building {brain_type} Tensor"):
            df_sym = fetch_yfinance_data(sym)
            if df_sym.empty:
                continue
            try:
                phys = generate_heavy_physics(df_sym)

                if brain_type == 'DIRECTION':
                    # Binary: did price close higher LOOK_FWD bars from now?
                    target = generate_direction_target(df_sym, look_fwd=look_fwd)

                else:  # EXP
                    # Range of bar N days forward, normalised by 14-bar avg range.
                    # > 1 = expanding, < 1 = contracting vs recent norm.
                    next_range = (df_sym['high'].shift(-look_fwd)
                                  - df_sym['low'].shift(-look_fwd))
                    avg_range  = (df_sym['high'] - df_sym['low']).rolling(14).mean()
                    target     = next_range / (avg_range + 1e-9)

                target.name = 'target'
                aligned = phys.join(target, how='inner').dropna()
                if len(aligned) > 50:
                    combined.append(aligned)
            except Exception as e:
                print(f"  ⚠ {brain_type} {sym}: {e}")

    if not combined:
        return pd.DataFrame()
    return pd.concat(combined, axis=0, ignore_index=True)

# =============================================================================
# AUDIT ENGINE — Train model + permutation importance
# =============================================================================
SEQ_LEN = 20

def run_judicial_audit(brain_type, master_df, model_type='GRU',
                       seq_len=SEQ_LEN, epochs=30, batch_size=512):
    """
    Trains one model on (seq_len, n_features) sequences, then scores each
    feature by permutation importance on the validation split.
    DIRECTION : importance = accuracy DROP  (baseline - permuted, >= 0)
    EASE/EXP  : importance = MAE RISE      (permuted - baseline,  >= 0)
    Returns DataFrame: Feature|I_raw|I_Norm|UV%|Max_R|Is_Locked|Persistence
    """
    target_col  = 'ease_val' if brain_type == 'EASE' else 'target'
    is_classify = (brain_type == 'DIRECTION')

    feature_cols = [c for c in master_df.columns if c != target_col]
    X = master_df[feature_cols].values.astype(np.float32)
    y = master_df[target_col].values.astype(np.float32)
    n = len(X)

    if n - seq_len < 100:
        print(f"  ⚠ Insufficient rows ({n}) for sequences — skipping.")
        return pd.DataFrame()

    train_n  = int(n * 0.70)
    scaler   = RobustScaler()
    scaler.fit(X[:train_n])
    X_scaled = scaler.transform(X).astype(np.float32)

    X_seqs = np.stack([X_scaled[i - seq_len:i] for i in range(seq_len, n)])
    y_seqs = y[seq_len:]
    n_seq  = len(X_seqs)

    train_end = int(n_seq * 0.70)
    val_end   = int(n_seq * 0.85)

    X_tr,  y_tr  = X_seqs[:train_end],        y_seqs[:train_end]
    X_val, y_val = X_seqs[train_end:val_end],  y_seqs[train_end:val_end]

    n_feat = len(feature_cols)
    print(f"  [DATA] train={len(X_tr):,}  val={len(X_val):,}  features={n_feat}")

    with tf.device(DEVICE):
        rec1 = GRU(64,  return_sequences=True) if model_type == 'GRU' else LSTM(64, return_sequences=True)
        rec2 = GRU(32) if model_type == 'GRU' else LSTM(32)
        model = Sequential([
            Input(shape=(seq_len, n_feat)),
            rec1, Dropout(0.2),
            rec2, Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1,  activation='sigmoid' if is_classify else 'linear'),
        ])
        model.compile(
            optimizer=Adam(3e-4),
            loss=('binary_crossentropy' if is_classify else tf.keras.losses.Huber()),
            metrics=(['accuracy'] if is_classify else ['mae']),
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True)],
            verbose=0,
        )

    # ── Baseline metric & quality gate ───────────────────────────────────────
    if is_classify:
        preds    = (model.predict(X_val, verbose=0).flatten() >= 0.5).astype(int)
        baseline = float(np.mean(preds == y_val.astype(int)))
        print(f"  [MODEL] Val Accuracy: {baseline:.4f}")
        if baseline <= 0.50:
            print("  ⚠ Accuracy gate FAILED — skipping.")
            del model; gc.collect(); tf.keras.backend.clear_session()
            return pd.DataFrame()
    else:
        preds    = model.predict(X_val, verbose=0).flatten()
        baseline = float(np.mean(np.abs(preds - y_val)))
        naive    = float(np.mean(np.abs(y_val)))
        print(f"  [MODEL] Val MAE: {baseline:.6f}  naive: {naive:.6f}")
        if baseline >= naive:
            print("  ⚠ MAE gate FAILED — model not better than naive. Skipping.")
            del model; gc.collect(); tf.keras.backend.clear_session()
            return pd.DataFrame()

    # ── Permutation importance ────────────────────────────────────────────────
    val_raw_slice = X_scaled[train_end : val_end + seq_len]
    n_val_rows    = val_end - train_end

    rows = []
    for fi, feat_name in enumerate(tqdm(feature_cols, desc=f"Permuting {brain_type}")):
        X_perm        = val_raw_slice.copy()
        flat          = X_perm[:, fi].copy()
        np.random.shuffle(flat)
        X_perm[:, fi] = flat
        X_perm_seqs   = np.stack([X_perm[k:k + seq_len] for k in range(n_val_rows)])

        if is_classify:
            p_perm = (model.predict(X_perm_seqs, verbose=0).flatten() >= 0.5).astype(int)
            m_perm = float(np.mean(p_perm == y_val.astype(int)))
            I_raw  = max(0.0, baseline - m_perm)
        else:
            p_perm = model.predict(X_perm_seqs, verbose=0).flatten()
            m_perm = float(np.mean(np.abs(p_perm - y_val)))
            I_raw  = max(0.0, m_perm - baseline)

        rows.append({'Feature': feat_name, 'I_raw': I_raw})

    del model; gc.collect(); tf.keras.backend.clear_session()

    report       = pd.DataFrame(rows)
    I_max        = report['I_raw'].max()
    report['I_Norm'] = report['I_raw'] / (I_max + 1e-9)

    corr     = master_df[feature_cols].corr().abs()
    uv_list, maxr_list = [], []
    for feat in feature_cols:
        others = [f for f in feature_cols if f != feat]
        uv_list.append((1.0 - corr[feat].loc[others].mean()) * 100 if others else 100.0)
        maxr_list.append(float(corr[feat].loc[others].max()) if others else 0.0)

    report['UV%']         = uv_list
    report['Max_R']       = maxr_list
    report['Is_Locked']   = False
    report['Persistence'] = 1
    return report

# =============================================================================
# EXPORT ENGINE — Single master CSV
# =============================================================================
_LEDGER_RE = re.compile(r"^LENS_(\d+)_(.+?)_(z|z_slope|z_sos)$")

def export_titan_ledger(brain_type, model_type, iteration, look_fwd,
                        report_df, feature_registry):
    """
    Appends one iteration's results to MASTER_CSV.
    feature_registry: dict {feature_name: cumulative_appearances} — updated in-place.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for feat in report_df['Feature']:
        feature_registry[feat] = feature_registry.get(feat, 0) + 1

    rows = []
    for _, row in report_df.iterrows():
        f_name = row['Feature']
        m      = _LEDGER_RE.match(f_name)
        if m:
            l_len, family, l_type = m.group(1), m.group(2), m.group(3)
        else:
            l_len, family, l_type = 'N/A', f_name, 'Raw'

        density = round(feature_registry.get(f_name, 0) / iteration, 2)

        rows.append({
            'Feature':          f_name,
            'I_raw':            row.get('I_raw',        0),
            'I_Norm':           row.get('I_Norm',       0),
            'UV%':              row.get('UV%',           0),
            'Max_R':            row.get('Max_R',         0),
            'Is_Locked':        row.get('Is_Locked',    False),
            'Iteration':        iteration,
            'Brain':            brain_type,
            'Model_Type':       model_type,
            'Look_Fwd':         look_fwd,
            'Timestamp':        timestamp,
            'Persistence':      row.get('Persistence',  0),
            'A_Impact':         row.get('I_raw',        0),
            'A_UV':             row.get('UV%',           0),
            'Family':           family,
            'Lens_Length':      l_len,
            'Lens_Type':        l_type,
            'Presence_Density': density,
        })

    out          = pd.DataFrame(rows)
    write_header = not os.path.exists(MASTER_CSV)
    out.to_csv(MASTER_CSV, mode='a', index=False, header=write_header)
    print(f"  ✅ {os.path.basename(MASTER_CSV)} | {brain_type} iter={iteration} "
          f"look_fwd={look_fwd} | {len(rows)} features appended")

# =============================================================================
# COMMAND CENTER
# =============================================================================
print("\n--- 🧠 SOVEREIGN TITAN v3.17.9 CONTROL PANEL ---")
brain_choice   = input("Select Brain (1:DIR / 2:EASE / 3:EXP / 4:ALL): ")
BRAINS_TO_RUN  = (['DIRECTION','EASE','EXP'] if brain_choice == '4'
                  else [{'1':'DIRECTION','2':'EASE','3':'EXP'}[brain_choice]])
num_symbols    = int(input("Symbols per iteration (Default 197): ") or "197")
num_iterations = int(input("Iterations to run (Default 10): ")      or "10")
LOOK_FWD       = int(input("Look-forward days (Default 1): ")       or "1")

print(f"\nPROTOCOL: Full mode | look_fwd={LOOK_FWD} | Output → {MASTER_CSV}")
print(f"  DIR  target : close[+{LOOK_FWD}] > close[0]  (binary classification)")
print(f"  EXP  target : range[+{LOOK_FWD}] / avg_range_14  (regression ratio)")
print(f"  EASE target : ease_val from parquet  (regression, no shift)")

feature_registry = {}   # global persistence tracker — all brains + iterations

for BRAIN in BRAINS_TO_RUN:
    MODEL_TYPE = 'GRU' if BRAIN == 'DIRECTION' else 'LSTM'
    print(f"\n{'═'*60}")
    print(f"  BRAIN: {BRAIN} | Model: {MODEL_TYPE} | look_fwd={LOOK_FWD}")
    print(f"{'═'*60}")

    for it in range(1, num_iterations + 1):
        POOL = random.sample(TITAN_SYMBOLS, min(num_symbols, len(TITAN_SYMBOLS)))
        print(f"\n🚀 {BRAIN} | Iteration {it}/{num_iterations}")
        try:
            master_df = load_hybrid_data(BRAIN, DRIVE_DB_PATH, POOL,
                                         look_fwd=LOOK_FWD)
            if master_df.empty:
                print("  ⚠ Empty tensor — skipping.")
                continue

            report = run_judicial_audit(BRAIN, master_df, model_type=MODEL_TYPE)
            if report.empty:
                print("  ⚠ Quality gate failed — skipping.")
                continue

            export_titan_ledger(BRAIN, MODEL_TYPE, it, LOOK_FWD,
                                report, feature_registry)
            print(f"  Tensor: {master_df.shape}")

        except Exception as e:
            print(f"  ❌ {BRAIN} iter {it}: {e}")
        finally:
            gc.collect()
            tf.keras.backend.clear_session()

print(f"\n{'═'*60}")
print(f"✅ AUDIT COMPLETE → {MASTER_CSV}")
print(f"{'═'*60}")
