# SOVEREIGN TITAN v4.0 — MASTER DOCUMENTATION
# ML-Based Financial Prediction System
# Physics-First Architecture | Trend-Only Indicators | Multi-Brain Framework

---

## TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Core Philosophy](#core-philosophy)
3. [Architecture Design](#architecture-design)
4. [Feature Engineering](#feature-engineering)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Model Configuration](#model-configuration)
7. [Anti-Leakage Safeguards](#anti-leakage-safeguards)
8. [Performance Optimization](#performance-optimization)
9. [Feature Importance System](#feature-importance-system)
10. [Quality Gates & Validation](#quality-gates--validation)
11. [Production Deployment](#production-deployment)
12. [Research Foundations](#research-foundations)

---

## PROJECT OVERVIEW

### Mission Statement
Build a production-grade machine learning system that predicts financial market movements with institutional-quality accuracy (0.014 MAE benchmark) by capturing the "Physics of the Move" through temporal dependencies and regime-aware feature engineering.

### Two-Stage Architecture (S1SFT → S2MFT)

#### Stage 1: Same Family Test (S1SFT)
**Goal:** Optimize within each feature domain independently

```
Single Feature Family (e.g., TREND)
├─ Start with ~30-50 indicator seeds
├─ Apply multi-lens transforms (windows, z-scores, slopes, ratios)
├─ Generate 100-200+ raw features (all same domain)
├─ PCA/RFE reduction: 100-200+ → 19 champions
└─ Output: Top 19 features representing that family
```

**Why 19 in Stage 1:**
- Aggressive within-family pruning
- Forces selection of only highest-signal indicators
- Prevents redundancy within domain
- Each family contributes equal weight (19 features) to Stage 2

**Stage 1 Families (Planned):**
1. **TREND** (current v4.0) — Price-based position, smoothing, ratios
2. **VOLUME** (planned) — Flow, accumulation, money flow
3. **MOMENTUM** (planned) — Oscillators, rate of change, relative strength
4. **VOLATILITY** (planned) — Range, ATR, Bollinger, Keltner
5. **OTHER** (planned) — Sentiment, fundamentals, alternative data

---

#### Stage 2: Multi-Family Test (S2MFT)
**Goal:** Combine best features across ALL domains

```
Multi-Family Pool
├─ TREND champions:      19 features
├─ VOLUME champions:     19 features
├─ MOMENTUM champions:   19 features
├─ VOLATILITY champions: 19 features
├─ OTHER champions:      19 features
├─ Total pool:           ~95 features
├─ Final PCA squeeze:    95 → 35 principal components
└─ Output: 35 multi-domain components (institutional benchmark)
```

**Why 35 in Stage 2:**
> "Submit final pooled tensor to PCA squeeze to reach top 35 principal components, retaining 95% of total variance, satisfying institutional benchmark for achieving 0.014 MAE."

**Brain-Specific Optimization:**
> Stage 1 → Stage 2 is executed SEPARATELY for each brain (DIRECTION, EASE, EXP). Different brains may select different champion features from the same families.

---

#### Current Implementation Status

**v4.0 = Stage 1 (S1SFT) for TREND Family:**
```
TREND Family:
├─ 39 indicator seeds (all trend-only)
├─ Multi-lens application → 222 raw features
├─ PCA reduction: 222 → 19 TREND champions
└─ Status: ✅ Complete and operational
```

**Planned Development:**

Phase 2 — Complete S1SFT for All Families:
- VOLUME family → 19 volume champions
- MOMENTUM family → 19 momentum champions
- VOLATILITY family → 19 volatility champions
- OTHER family → 19 other champions

Phase 3 — Implement S2MFT:
- Pool all family champions (~95 features)
- Final PCA: 95 → 35 components
- Train each brain on 35-component input
- Achieve 0.014 MAE benchmark

---

### Visual Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SOVEREIGN TITAN ARCHITECTURE                      │
│                  Two-Stage Feature Selection Workflow                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: SAME FAMILY TEST (S1SFT)                                  │
│  Goal: Optimize within each domain independently                    │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │   TREND      │      │   VOLUME     │      │  MOMENTUM    │
    │   Family     │      │   Family     │      │   Family     │
    └──────────────┘      └──────────────┘      └──────────────┘
          │                      │                      │
    30-50 seeds            30-50 seeds            30-50 seeds
          │                      │                      │
    Apply lenses           Apply lenses           Apply lenses
    (windows, z,           (windows, z,           (windows, z,
     slope, sos,            slope, sos,            slope, sos,
     ratios)                ratios)                ratios)
          │                      │                      │
          ▼                      ▼                      ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  100-200+    │      │  100-200+    │      │  100-200+    │
    │  TREND       │      │  VOLUME      │      │  MOMENTUM    │
    │  features    │      │  features    │      │  features    │
    └──────────────┘      └──────────────┘      └──────────────┘
          │                      │                      │
      PCA / RFE              PCA / RFE              PCA / RFE
      222 → 19               150 → 19               180 → 19
          │                      │                      │
          ▼                      ▼                      ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │ 19 TREND     │      │ 19 VOLUME    │      │ 19 MOMENTUM  │
    │ CHAMPIONS    │      │ CHAMPIONS    │      │ CHAMPIONS    │
    └──────────────┘      └──────────────┘      └──────────────┘

    ┌──────────────┐      ┌──────────────┐
    │ VOLATILITY   │      │    OTHER     │
    │   Family     │      │   Family     │
    └──────────────┘      └──────────────┘
          │                      │
    30-50 seeds            20-40 seeds
          │                      │
    Apply lenses           Apply lenses
          │                      │
          ▼                      ▼
    ┌──────────────┐      ┌──────────────┐
    │  100-200+    │      │  80-150+     │
    │ VOLATILITY   │      │   OTHER      │
    │  features    │      │  features    │
    └──────────────┘      └──────────────┘
          │                      │
      PCA / RFE              PCA / RFE
      175 → 19               120 → 19
          │                      │
          ▼                      ▼
    ┌──────────────┐      ┌──────────────┐
    │ 19 VOLATILITY│      │  19 OTHER    │
    │  CHAMPIONS   │      │  CHAMPIONS   │
    └──────────────┘      └──────────────┘
          │                      │
          └──────────┬───────────┘
                     │
                     ▼

┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: MULTI-FAMILY TEST (S2MFT)                                 │
│  Goal: Combine best features across ALL domains                     │
└─────────────────────────────────────────────────────────────────────┘

            ┌─────────────────────────────────┐
            │   MULTI-FAMILY POOL             │
            │                                 │
            │  TREND:      19 features        │
            │  VOLUME:     19 features        │
            │  MOMENTUM:   19 features        │
            │  VOLATILITY: 19 features        │
            │  OTHER:      19 features        │
            │  ─────────────────────          │
            │  TOTAL:      ~95 features       │
            └─────────────────────────────────┘
                         │
                    FINAL PCA
                    95 → 35
                         │
                         ▼
            ┌─────────────────────────────────┐
            │   35 PRINCIPAL COMPONENTS       │
            │   (Multi-Domain Features)       │
            │                                 │
            │   - 95% total variance retained │
            │   - Cross-family optimization   │
            │   - Institutional benchmark     │
            └─────────────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────┐
            │    BRAIN-SPECIFIC MODELS        │
            │                                 │
            │  ┌─────────────────────────┐   │
            │  │  DIRECTION Brain (GRU)  │   │
            │  │  Input: 35 components   │   │
            │  │  Target: 58-65% acc     │   │
            │  └─────────────────────────┘   │
            │                                 │
            │  ┌─────────────────────────┐   │
            │  │  EASE Brain (LSTM)      │   │
            │  │  Input: 35 components   │   │
            │  │  Target: 0.014 MAE      │   │
            │  └─────────────────────────┘   │
            │                                 │
            │  ┌─────────────────────────┐   │
            │  │  EXP Brain (LSTM)       │   │
            │  │  Input: 35 components   │   │
            │  │  Target: 0.015 MAE      │   │
            │  └─────────────────────────┘   │
            └─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CURRENT STATUS (v4.0)                                              │
└─────────────────────────────────────────────────────────────────────┘
    ✅ S1SFT for TREND Family COMPLETE
       - 39 seeds → 222 features → 19 champions

    ⚠️  S1SFT for VOLUME Family     - PLANNED (Phase 2)
    ⚠️  S1SFT for MOMENTUM Family   - PLANNED (Phase 2)
    ⚠️  S1SFT for VOLATILITY Family - PLANNED (Phase 2)
    ⚠️  S1SFT for OTHER Family      - PLANNED (Phase 2)

    ⚠️  S2MFT Multi-Family Integration - PLANNED (Phase 3)
       - 95 champions → 35 final components
       - Target: 0.014 MAE benchmark
```

---

### Primary Goals

1. **Directional Prediction (DIRECTION Brain)**
   - Binary classification: Will tomorrow's close exceed today's close?
   - Target metric: 52-60% test accuracy (signal detection)
   - Quality gate: Reject if validation <52% (noise filter)

2. **Tradability Prediction (EASE Brain)**
   - Regression: Predict tomorrow's 3-minute bar tradability
   - Target metric: MAE approaching 0.014 (institutional benchmark)
   - Use case: Execution timing optimization

3. **Range Expansion Prediction (EXP Brain)**
   - Regression: Predict tomorrow's range relative to 20-day average
   - Target metric: Minimize MAE on normalized range
   - Use case: Volatility-based position sizing

### Success Criteria
- ✅ Directional accuracy >52% (demonstrates signal vs noise)
- ✅ No data leakage (test accuracy ≤ validation accuracy)
- ✅ Reproducible results (consistent across iterations)
- ✅ Scalable architecture (20-200 symbols)
- ✅ Production-ready (CPU/GPU adaptive, error handling)

---

## CORE PHILOSOPHY

### Physics-First Approach

> "Markets exhibit physical properties: position, velocity, acceleration. Capturing these temporal dynamics is essential for predictive alpha."

- **Level 2 (Position):** Z-score normalization captures where price sits in regime
- **Level 3 (Velocity):** Slope of Z-score measures trend strength
- **Level 4 (Acceleration):** Slope-of-slope detects momentum turning points

### Trend-Only Mandate

- ✅ TREND indicators only (price-based, position, smoothing)
- ❌ NO VOLUME (introduces different data frequency artifacts)
- ❌ NO VOLATILITY (implicit in range-based but not explicit)
- ❌ NO MOMENTUM oscillators (RSI, Stochastic, Williams %R)

### Curation Over Volume

> "More features ≠ Better model. Curated, orthogonal features > Kitchen sink."

- Start with 39 indicator seeds (carefully selected)
- Apply multi-order lenses (6 transforms per Z_LENS seed)
- PCA reduces 222→19 components (retains 95% variance)
- Feature importance identifies weak performers for removal

---

## ARCHITECTURE DESIGN

### Three-Brain Framework

#### DIRECTION Brain
```python
Architecture:    GRU (faster than LSTM for classification)
Activation:      Sigmoid (binary output 0-1)
Loss:            Binary Crossentropy
Data Window:     6 years
Sequence Length: 30 days (Fast Mode) / 60 days (Full Mode)
Target:          (tomorrow_close > today_close).astype(int)
```

#### EASE Brain
```python
Architecture:    LSTM (better for regression with external data)
Activation:      Linear (unbounded regression output)
Loss:            Huber (robust to outliers/market shocks)
Data Window:     4 years
Sequence Length: 30 days (Fast Mode) / 60 days (Full Mode)
Target:          ease_parquet['EASE_val'].shift(-1)
```

#### EXP Brain
```python
Architecture:    LSTM (regression on normalized range)
Activation:      Linear
Loss:            Huber
Data Window:     4 years
Sequence Length: 30 days (Fast Mode) / 60 days (Full Mode)
Target:          tomorrow_range / rolling_mean(range, 20)
```

### Model Architecture Details

**Fast Mode (Default — 3x Faster):**
```python
Input:   (batch, 30, 19)
Layer 1: GRU/LSTM(64, return_sequences=True)
Layer 2: Dropout(0.2)
Layer 3: GRU/LSTM(32)
Layer 4: Dropout(0.2)
Layer 5: Dense(16, activation='relu')
Layer 6: Dense(1, activation=output_activation)
Total Parameters: ~25,000
```

**Full Mode (Maximum Accuracy):**
```python
Input:   (batch, 60, 19)
Layer 1: GRU/LSTM(128, return_sequences=True)
Layer 2: Dropout(0.2)
Layer 3: GRU/LSTM(64)
Layer 4: Dropout(0.2)
Layer 5: Dense(32, activation='relu')
Layer 6: Dense(1, activation=output_activation)
Total Parameters: ~100,000
```

---

## CURRENT FEATURE IMPLEMENTATIONS (S1SFT)

### TREND Family — COMPLETE (v4.0) ✅

**Status:** Production-ready, 37 z_lens seeds + 3 win seeds = 222 features → 19 champions

#### Numba Kernel Architecture

All indicator calculations use JIT-compiled Numba kernels for 10-50x speedup:

```python
@jit(nopython=True, cache=True, fastmath=True)
```

**14 Compiled Kernels:**
1. `_fast_linslope` — Linear regression slope (closed-form)
2. `_fast_zscore` — Rolling z-score normalization
3. `_fast_shannon` — Shannon entropy (10 bins)
4. `_fast_hurst` — Hurst exponent via rescaled range
5. `_fast_tema` — Triple exponential moving average
6. `_fast_r_sq` — R-squared from linear regression
7. `_fast_wma` — Weighted moving average
8. `_fast_fwma` — Fibonacci weighted moving average
9. `_fast_polynomial_fit` — 2nd order polynomial (velocity/acceleration)
10. `_fast_mass_index` — Mass Index (EMA ratio sum)
11. `_kalman_numba` — Kalman filter (r=0.0001, q=0.001)
12. `_rolling_cog` — Center of Gravity (Ehlers)
13. `_vhf_nb` — Vertical Horizontal Filter
14. `_psar_nb` — Parabolic SAR with trend

**Warmup Routine:** All kernels pre-compiled on startup with dummy data.

---

#### 37 Z_LENS Seeds (Stationary Oscillators/Ratios)

**Treatment:** Each gets 6 transforms (10d/90d × z/slope/sos) = 222 features

| Category | Seeds | Notes |
|---|---|---|
| Stationarity Foundation | `log_diff` | log(price_t) - log(price_t-1) |
| Interaction Ratios | `tema_30_ratio`, `tema_10_ratio`, `sma_5_ratio`, `sma_20_ratio`, `vidya_10_ratio`, `kalman_ratio` | Pre-digest non-linear behavior |
| VWAP-Based | `mtsi` | EMA(EMA(Close - VWAP, 3), 2) |
| Efficiency Estimators | `er_20`, `vidya_cmo` | Trend vs noise measurement |
| Fractal Efficiency | `hurst_50`, `hurst_20` | H>0.5: persistent, H<0.5: mean-reverting |
| Entropy | `shannon_20`, `shannon_10` | High=disordered, Low=structured |
| Trend Quality | `r_sq_30`, `r_sq_10`, `linreg_slope_30` | Linearity measurement |
| Trend Strength | `adx_14` | ADX>25: strong, ADX<20: weak |
| Center of Gravity | `cog_20` | Ehlers; Z_LENS (not WIN) — locally stationary |
| Composite Ratios | `shannon_ratio_10_20`, `entropy_r_sq_ratio`, `r_sq_ratio_10_30`, `er_ratio_10_20`, `dispersion_r_sq` | Pre-digested interactions |
| Research Additions | `hma_21_ratio`, `fwma_13_ratio`, `medium_filter_ratio`, `price_velocity_60`, `price_acceleration_60`, `ema_50_200_ratio`, `trend_factor_10/30/100/200`, `mass_index_25`, `ma_distance_60_std` | 13 academic-validated seeds |

#### 3 WIN Seeds (Unbounded Metrics)

**Treatment:** Each gets 2 transforms (10d/30d rolling %) = 6 features

| Seed | Formula | Rationale |
|---|---|---|
| `vhf_28` | (max - min) / Σ\|changes\| | Vertical Horizontal Filter |
| `psar_distance` | (Close - PSAR) / PSAR | Distance from Parabolic SAR |
| `psar_trend` | +1 or -1 | PSAR trend direction |

#### Feature Count Summary
```
Z_LENS: 37 seeds × 6 transforms = 222 features
WIN:     3 seeds × 2 transforms =   6 features
─────────────────────────────────────────────
TOTAL:  40 seeds                = 228 features
After PCA:                        19 TREND champions
```

---

### VOLUME Family — IN PROGRESS ⚠️

**Target:** ~20-25 seeds → ~100-150 features → 19 VOLUME champions

**Planned Seeds (Categories):**
1. Fundamental Volume Physics — liquidity_ratio, amivest, CMF, FI, rel_vol, TMF, hurst_vol
2. Path & Movement Efficiency — EMV, FVE, PVO, harlin_spike, harlin_osc, mobius_bsp, vwap_dev
3. Composite Ratios — force/friction, vel/friction, inst_squeeze v1/v2, kaufman_vol_hybrid
4. Cumulative (WIN treatment) — PVT, OBV, OBV oscillator

---

### MOMENTUM Family — IN PROGRESS ⚠️

**Target:** ~20-25 seeds → ~100-150 features → 19 MOMENTUM champions

**Planned Seeds (Categories):**
1. Velocity & Acceleration (Physics) — v_log, v_roc2, accel, froude, g_force
2. Efficiency & Entropy — er_20, shannon_momentum, scr_20
3. DeMark Sequential — td_buy, td_sell
4. GDT Ease of Capture — gdt_ease

---

### VOLATILITY Family — PLANNED ⚠️

**Target:** ~15-20 seeds → ~90-120 features → 19 VOLATILITY champions

Planned: ATR variants, Bollinger Bands, Keltner Channels, standard deviation, Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang estimators, volatility ratios.

---

### OTHER Family — PLANNED ⚠️

**Target:** ~15-20 seeds → ~90-120 features → 19 OTHER champions

Planned: FinBERT sentiment, Put/Call ratios, VIX-related, market breadth, sector rotation, macroeconomic, intermarket, options-implied metrics.

---

## TESTING FRAMEWORK EVOLUTION

### v3.19.18: Judicial Audit + Sovereign Hunt

**Feature Selection: "Sovereign Hunt" (Permutation Importance)**
- Single model trained once per iteration
- Permutation importance on validation set (150+ passes)
- Diversity enforcement: correlation threshold 0.85, UV% uniqueness scoring
- Family-lookback rule: only one window per indicator
- Persistence tracking across iterations (A_Impact, A_UV)

**Advantages:** True predictive utility, diversity enforcement, persistence tracking
**Disadvantages:** Slower (150 permutation passes), complex hunt logic

---

### v4.0: Streamlined PCA + Fast/Importance Modes

**Feature Selection: "PCA Reduction"**
- Scale features (fit on train only)
- PCA reduction (fit on train only)
- Feature importance from PCA loadings aggregated to seed level
- Fast Mode (3-5x speedup), Importance-Only Mode (~30 sec screening)

**Advantages:** Massive seed expansion (39 vs 16), 3-5x faster, simpler logic, works everywhere
**Disadvantages:** PCA shows variance not alpha, no diversity enforcement, no persistence tracking

---

### Shared Testing Methodology

**Walk-Forward Splits:**
```python
train_end = int(len(X) * 0.70)   # 70%
val_end   = int(len(X) * 0.85)   # 15%
                                   # 15% test
```

**Anti-Leakage:** Scaler and PCA fit on train only, warmup rows (350) excluded.

**Quality Gates:**
```python
# DIRECTION
if val_accuracy < 0.50: skip  # No signal
if val_accuracy > 0.70 and test_accuracy > 0.70: investigate_leakage

# EASE/EXP: Track MAE, target 0.014-0.020
```

**BRAIN_LOCKS:**
```python
BRAIN_LOCKS = {
    'DIRECTION': ['hma_21_ratio_z', 'shannon_ratio_10_20_z',
                  'fwma_13_ratio_z', 'entropy_r_sq_ratio_z'],
    'EASE': [],
    'EXP':  ['LENS_10_hurst_50_z', 'LENS_90_cog_20_z_sos']
}
```

**Recommended Hybrid Approach (Future):**
- Phase 1: Rapid screening (v4.0 Importance-Only, ~3 min) → prune bottom 15 seeds
- Phase 2: Deep validation (v3.19.18 Sovereign Hunt) → 10 iterations for persistence
- Phase 3: Lock champions, final validation

---

### Key Differences

| Aspect | v3.19.18 Judicial Audit | v4.0 PCA Reduction |
|---|---|---|
| Method | Permutation importance | PCA variance |
| What it measures | Predictive utility (alpha) | Data variance |
| Diversity control | Correlation guards (0.85) | None |
| Speed | Slower (150 permutations) | 3-5x faster |
| Feature count | 93 → 19 | 228 → 19 |

---

## FEATURE ENGINEERING

### Z_LENS Indicators (Stationary Oscillators/Ratios)

**6 transforms per seed** (for windows W ∈ {10, 90}):
```python
1. Z-score:           (S - rolling_mean(S, W)) / rolling_std(S, W)
2. Z-slope:           linear_regression_slope(Z-score, W)
3. Z-slope-of-slope:  linear_regression_slope(Z-slope, W)
# Result: 6 features per seed
```

### WIN Indicators (Unbounded Metrics)

**2 transforms per seed** (for windows W ∈ {10, 30}):
```python
Rolling_pct = S / rolling_mean(S, W) - 1
# Result: 2 features per seed
```

### Key Indicator Formulas

**TEMA:** `3*EMA(n) - 3*EMA(EMA(n)) + EMA(EMA(EMA(n)))`
**VIDYA:** `alpha * |CMO| * price + (1 - alpha * |CMO|) * VIDYA[t-1]`
**Kalman:** `r=0.0001 (measurement noise), q=0.001 (process noise)`
**HMA:** `WMA(2*WMA(n/2) - WMA(n), √n)`
**FWMA:** `Σ(Close_i * F_i) / Σ(F_i)` using Fibonacci weights
**ER:** `abs(close - close.shift(20)) / sum(abs(close.diff()), 20)`
**COG:** `-Σ(w_i * price_i) / Σ(price_i)` (Ehlers)
**PSAR:** `PSAR[i] = PSAR[i-1] + AF * (EP - PSAR[i-1])`

**Physics Seeds (Critical):**
```python
price_velocity_60:     A₁ from S(t) = A₀ + A₁×t + A₂×t²
price_acceleration_60: A₂ from polynomial fit
```
> "Velocity and acceleration provide the Physics of the Move required for LSTMs to detect turning points before they manifest in price."

---

## DATA PROCESSING PIPELINE

### 9-Step Workflow

**Step 1 — Data Acquisition:**
```python
Source: yfinance | Period: 6y (DIR), 4y (EASE/EXP)
Quality Gate: Minimum 400 raw rows (250 in Fast Mode)
```

**Step 2 — Feature Calculation:**
- 39 indicator seeds → 222 feature columns via Numba kernels

**Step 3 — NaN Handling (Smart Multi-Stage):**
```python
# Drop only if >50% of features are NaN
df = df.dropna(subset=feature_cols, thresh=len(feature_cols) - int(nan_threshold))
df[feature_cols] = df[feature_cols].ffill()   # Forward-fill (no leakage)
df[feature_cols] = df[feature_cols].fillna(0) # Zero-fill residual
# Result: ~80% data survival vs 0% with naive dropna
```

**Step 4 — Target Definition:**
```python
DIRECTION: (close.shift(-1) > close).astype(int)
EASE:      ease_parquet['EASE_val'].shift(-1)   # or fallback normalized move
EXP:       (high - low).shift(-1) / rolling_mean(high - low, 20)
```

**Step 5 — Concatenation:** Multi-symbol master dataframe (~10,000-20,000 rows)

**Step 6 — Train/Val/Test Split:** Walk-forward, 70/15/15

**Step 7 — Sequence Creation:**
```python
seq_len = 30  # Fast Mode (60 in Full Mode)
# Overlapping windows: row i uses days [i-30:i]
```

**Step 8 — Scaling (RobustScaler):**
```python
scaler = RobustScaler()
scaler.fit(X_train_2d[warmup:])  # CRITICAL: train only, exclude warmup
```

**Step 9 — PCA Dimensionality Reduction:**
```python
pca = PCA(n_components=19)
pca.fit(X_train_flat[warmup:])   # CRITICAL: train only
# Typical variance explained: 95-100%
```

---

## MODEL CONFIGURATION

### Training Parameters

| Parameter | Fast Mode | Full Mode |
|---|---|---|
| Epochs | 20 | 50 |
| Batch Size | 2048 | 512 |
| Seq Length | 30 | 60 |
| Model Units | [64, 32, 16] | [128, 64, 32] |
| Early Stop Patience | 5 | 10 |
| LR Reduce Patience | 3 | 5 |
| Time/Iteration | 2-4 min | 8-12 min |

### Callbacks

```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
Adam(learning_rate=1e-3)
```

---

## ANTI-LEAKAGE SAFEGUARDS

### 5 Critical Protections

**1. Walk-Forward Validation** — Chronological split, no shuffle
**2. Fit Transforms on Training Only** — Scaler and PCA
**3. Warmup Exclusion** — First 350 rows excluded from fit
**4. No Future Data in Features** — No `.shift(-1)` in feature columns
**5. Target Isolation** — Last row target = NaN

### Leakage Detection
```python
# Healthy
val=0.54, test=0.52  # ✅ Expected slight overfit
val=0.60, test=0.58  # ✅ Good signal

# Leakage
val=0.55, test=0.75  # 🚨 Test >> Val
val=0.88, test=0.89  # 🚨 Both too high (identity or leakage)
```

### Identity Function Trap Prevention ✅
```python
# CORRECT
T_FINAL = close.shift(-1)          # Tomorrow's close as target

# WRONG — causes 90% "accuracy" (useless)
T_FINAL = close                     # ❌ Identity function trap
features['close_lag_1'] = close.shift(1)  # ❌ Future leakage
```

---

## PERFORMANCE OPTIMIZATION

### CPU/GPU Adaptive Execution
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Numba JIT Compilation
```python
@jit(nopython=True, cache=True, fastmath=True)
# Speedup: linear slope ~50x, Hurst ~30x, Shannon ~20x
# Overall: 10-30x faster feature calculation
```

### Memory Management
```python
del model; gc.collect(); tf.keras.backend.clear_session()
```

---

## FEATURE IMPORTANCE SYSTEM

### PCA-Based Importance (Current)
```python
loadings = pca.components_.T           # (222 features, 19 components)
importance = np.abs(loadings).sum(axis=1)
# Aggregate to seeds, sort descending
```

### Critical Distinction
> "PCA importance identifies noise to prune (pre-training). Training importance identifies alpha to extract (post-training)."

| Method | Measures | Use Case |
|---|---|---|
| PCA Importance | Variance contribution | Pre-training noise screening |
| Permutation Importance | Prediction degradation | True predictive utility |
| SHAP Values | Game-theoretic attribution | Explainability, debugging |

**Planned:** SHAP + Permutation Importance for Phase 1.5

---

## QUALITY GATES & VALIDATION

### Expected Performance Ranges

| Brain | Excellent | Acceptable | Poor |
|---|---|---|---|
| DIRECTION | 55-65% accuracy | 52-55% | <52% (reject) |
| EASE MAE | <0.014 | 0.014-0.025 | >0.025 |
| EXP MAE | <0.015 | 0.015-0.030 | >0.030 |

### Quality Gate Code
```python
# DIRECTION: reject if below noise floor
if val_accuracy < 0.52:
    continue

# Leakage: investigate if both too high
if test_accuracy > 0.70 and val_accuracy > 0.70:
    investigate_leakage()
```

---

## ADVANCED BEST PRACTICES (10 Institutional Techniques)

### 1. CEEMDAN Decomposition — ⚠️ Future (Phase 2+)
Decompose price into high/medium/low frequency IMFs before modeling to reduce noise impact.

### 2. Pillar-First Functional Decomposition — ✅ Implemented
GRU for DIRECTION (binary, fast convergence), LSTM for EASE/EXP (long-range dependencies).

### 3. Bayesian Hyperparameter Optimization — ⚠️ Future (Phase 2+)
50-iteration Bayesian search over units, dropout, learning rate per brain.

### 4. SHAP + Permutation Importance — 🔄 Partial (Phase 1.5)
Replace/supplement PCA importance with true predictive utility measurement.

### 5. Online Changepoint Detection — ⚠️ Future (Phase 2+)
Gaussian Process with Matérn 3/2 kernel to detect regime shifts and quantify disequilibrium.

### 6. Sentiment Risk Shield — ⚠️ Future (Phase 3+)
FinBERT-derived sentiment as gating mechanism. Block long trades if sentiment < -0.70.

### 7. CNN-LSTM Hybrid Architecture — ⚠️ Future (Phase 2+)
```python
Input → [Conv1D(causal padding)] → [LSTM/GRU] → Output
# CNN extracts spatial chart patterns before temporal processing
```

### 8. Medium Filter Anchor — ✅ Implemented
`medium_filter_ratio = HLC3 / ((High + Low) / 2)` — smooths wicks, improves generalization.

### 9. Recursive Feature Elimination — ⚠️ Future (Phase 1.5+)
Iteratively remove weakest features and recalculate importance (vs static PCA).

### 10. The 35-Component Squeeze — ⚠️ Stage 2 Target (Phase 3)
Applies to S2MFT only. Pool 95 champions → PCA → 35 components (institutional benchmark).

---

## THE TRIPLE SCREEN PROTOCOL

```
Screen 1 (DIR):    P(direction) > 0.65
Screen 2 (EASE):   friction_index < 0.40
Screen 3 (EXPAND): P(expansion) > 0.55
```

> **Rule:** If any screen fails → No Trade. Standard Work: do routine things routinely.

---

## IMPLEMENTATION STATUS SUMMARY

### Stage 1 (S1SFT) Status

| Family | Seeds | Raw Features | Champions | Status |
|---|---|---|---|---|
| TREND | 39 | 222 | 19 | ✅ Complete (v4.0) |
| VOLUME | ~40 | ~150-200 | 19 | ⚠️ Phase 2 |
| MOMENTUM | ~40 | ~180-200 | 19 | ⚠️ Phase 2 |
| VOLATILITY | ~35 | ~175-200 | 19 | ⚠️ Phase 2 |
| OTHER | ~30 | ~120-150 | 19 | ⚠️ Phase 2 |

### Stage 2 (S2MFT) Status

| Component | Input | Output | Status |
|---|---|---|---|
| Family Pool | 5×19 = 95 | — | ⚠️ Phase 3 |
| Final PCA | 95 | 35 | ⚠️ Phase 3 |
| Brain Training | 35 | Predictions | ⚠️ Phase 3 |

### Best Practices Scorecard

| Practice | Status | Phase |
|---|---|---|
| Pillar-First (GRU/LSTM) | ✅ Complete | v4.0 |
| Medium Filter | ✅ Complete | v4.0 |
| Multi-Lens Hierarchy | ✅ Complete | v4.0 |
| 60-Day Lookback | ✅ Complete (Full Mode) | v4.0 |
| Stationarity Discipline | ✅ Complete | v4.0 |
| Identity Function Prevention | ✅ Complete | v4.0 |
| Consensus Indicators (HLC3/TEMA/FWMA) | ✅ 3/4 | v4.0 |
| 19-Component PCA (S1SFT) | ✅ Complete | v4.0 |
| SHAP/Permutation Importance | 🔄 Partial | 1.5 |
| Recursive Feature Elimination | ⚠️ Future | 1.5+ |
| Bayesian Optimization | ⚠️ Future | 2+ |
| Changepoint Detection | ⚠️ Future | 2+ |
| CNN-LSTM Hybrid | ⚠️ Future | 2+ |
| CEEMDAN Decomposition | ⚠️ Future | 2+ |
| Sentiment Risk Shield | ⚠️ Future | 3+ |
| 35-Component PCA (S2MFT) | ⚠️ S2 Target | 3 |
| Triple Screen Protocol | ⚠️ Future | 3+ |

---

## PHASE ROADMAP

| Phase | Focus | Output | Target Metrics |
|---|---|---|---|
| **1 (Current — v4.0)** | S1SFT TREND family | 19 TREND champions | 52-58% DIR, 0.015-0.025 MAE |
| **1.5 (Near-Term)** | Optimize TREND S1SFT (SHAP, RFE) | Validated 19 champions | Confirm TREND baseline |
| **2 (Next Quarter)** | S1SFT all 4 remaining families | 5×19 = 95 champions | Full family coverage |
| **3 (Future)** | S2MFT multi-family + triple screen | 35 components/brain | 58-65% DIR, 0.014 MAE |
| **3+ (Advanced)** | Bayesian opt, CPD, CNN-LSTM, sentiment | Production system | Live trading robustness |

---

## RESEARCH FOUNDATIONS

### Key Principles

1. **Physics-First** — Position/Velocity/Acceleration hierarchy
2. **Stationarity Mandate** — Log-differencing; ratios over raw prices
3. **Curation Over Volume** — 19 curated > 100+ kitchen-sink
4. **Anti-Leakage** — Walk-forward, fit on train only
5. **Supervised Validation Required** — PCA prunes noise, training extracts alpha

### Institutional Benchmarks

| Metric | Poor | Acceptable | Good | Institutional |
|---|---|---|---|---|
| DIR Accuracy | <52% | 52-55% | 55-60% | 60%+ |
| EASE MAE | >0.040 | 0.025-0.040 | 0.015-0.025 | **0.014** |

### Key Research Citations
- Kinematic hierarchy: Position → Velocity → Acceleration for LSTM input
- Hurst: "Predictability positively correlated with Hurst exponent"
- Entropy: "Decreasing entropy confirms decreased market efficiency"
- Huber Loss: "Stabilizes training against market shocks"
- PCA: "35 components retaining 95% variance for institutional benchmark"
- Identity Trap: "One-day lag in target causes model to learn Tomorrow = Today"

---

## QUICK REFERENCE

### Architecture Decision Tree
```
Predicting binary (Up/Down)?  → GRU  (DIRECTION Brain)
Predicting continuous value?  → LSTM (EASE / EXP Brain)
```

### Key Numbers
```
19  = S1SFT target (within-family champions per brain)
35  = S2MFT target (cross-family final features per brain)
95  = Pool size (5 families × 19 champions)
95% = Variance retention threshold (both stages)
```

### Critical Do's ✅
1. GRU for DIRECTION, LSTM for EASE/EXP
2. `T_FINAL = close.shift(-1)` — never use lagged close as feature
3. Fit scaler and PCA on training data only
4. Walk-forward chronological splits
5. Include medium filter `(H+L)/2` for wick smoothing
6. Apply multi-order lenses (z, slope, sos)
7. Log-difference for stationarity
8. RobustScaler for fat-tailed financial distributions

### Critical Don'ts ❌
1. No future data in features
2. No scaler/PCA fit on full dataset
3. No random train/test split for time-series
4. Do not trust PCA importance alone (shows variance, not alpha)
5. Never skip early stopping
6. Never mix volume with trend-only S1SFT

### Common Pitfalls

| Pitfall | Symptom | Solution |
|---|---|---|
| Identity Function Trap | 90% accuracy, useless | `.shift(-1)` for target only |
| Data Leakage | Test > Val accuracy | Fit on train, walk-forward |
| No Convergence | Loss plateaus high | Reduce LR, check features |
| Overfitting | Val 60%, Test 45% | Add dropout, early stopping |
| All Data Deleted | "No data collected" | Fix NaN handling, lower min rows |
| Slow Training | >10 min/iteration | Enable FAST_MODE, Numba |

### Performance Expectations (v4.0 — TREND S1SFT)
```
DIRECTION:      52-58% test accuracy
EASE MAE:       0.015-0.025
EXP MAE:        0.018-0.030
Fast Mode time: 2-4 min/iteration
Full Mode time: 8-12 min/iteration
```

### Performance Targets (Phase 3 — Full S2MFT)
```
DIRECTION:      58-65% test accuracy
EASE MAE:       0.014-0.020  ← Institutional benchmark
EXP MAE:        0.015-0.025
```

---

**END OF MASTER DOCUMENTATION**

**Version:** 4.0 Complete with Two-Stage Architecture + Best Practices
**Last Updated:** 2026-03-08
**Status:** Phase 1 Complete (S1SFT TREND) ✅ | Phase 2-3 Planned
**Architecture:** S1SFT (19 champions/family) → S2MFT (95→35 multi-domain)
**Compliance:** 10/10 Best Practices identified | 8/10 implemented for S1SFT
