# SOVEREIGN TITAN — Master Architectural Reference (v2026.1)

> **Purpose:** This document is the Architectural North Star for the Sovereign Titan project.
> Every code block generated must follow the mechanical discipline and logic defined here.

---

## PROJECT OVERVIEW

**Goal:** Build a non-discretionary, multi-stage machine learning engine that identifies
1-day market opportunities by isolating:

1. **Directional Probability** — Is price likely to close higher or lower tomorrow?
2. **Ease of Movement** — How much resistance/friction exists in that direction?
3. **Volatility Expansion** — Will tomorrow be a range expansion or compression day?

**Target Benchmark:** 0.014 Mean Absolute Error (MAE) — institutional-grade performance.

---

## I. CORE ARCHITECTURAL GOALS

| Principle | Description |
|---|---|
| **Mechanical Discipline** | Eliminate discretionary brilliance in favor of Standard Work — repeatable, mathematical loops. |
| **Tri-Brain Synergy** | Three specialized neural networks: DIRECTION (GRU), EASE (LSTM), EXPAND (LSTM). |
| **Lookback Homogeneity** | Every indicator family locked to a single best-fit lookback to prevent conflicting timeframe signals. |
| **Horizon Locking** | All training/backtesting defaults to a 1-day lookback/prediction horizon. |
| **Full Execution** | Never generate simplified versions — always produce complete, production-ready code blocks. |

---

## II. TRI-BRAIN ARCHITECTURE

### Brain 1 — DIRECTION (GRU)
- **Question:** Is price likely to close higher or lower tomorrow?
- **Why GRU:** Simpler gating mechanism, superior for capturing binary directional nature with less overfitting risk.
- **Loss:** Binary Cross-Entropy (BCE)
- **Signal Threshold:** Probability > 0.65

### Brain 2 — EASE (LSTM)
- **Question:** How much resistance/friction exists in that direction?
- **Why LSTM:** Memory cells handle complex long-memory patterns in price friction and magnitude.
- **Loss:** Huber Loss (minimizes Black Swan outlier impact)
- **Signal Threshold:** Friction index < 0.40

### Brain 3 — EXPAND (LSTM)
- **Question:** Will tomorrow be a range expansion or compression day?
- **Why LSTM:** Long-range temporal dependencies in volatility require LSTM memory architecture.
- **Loss:** Huber Loss
- **Signal Threshold:** Expansion probability > 0.55

---

## III. THE TRIPLE SCREEN PROTOCOL

Trades are only suggested when **all three screens align**:

```
Screen 1 (DIR):    P(direction) > 0.65
Screen 2 (EASE):   friction_index < 0.40
Screen 3 (EXPAND): P(expansion) > 0.55
```

> **Rule:** If any screen fails → output is **No Trade**.
> Standard Work: do routine things routinely. Never force a trade.

---

## IV. DATA INTEGRITY & PRE-CONDITIONING

### Stationarity Mandate
Apply **logarithmic differencing** to all non-stationary price-level inputs and targets:
```
log(Open_t) - log(Open_{t-1})
```
Stabilizes chaotic dynamics and resolves non-stationary price levels.

### No-Lag Guardrail
Always predict the **non-lagged daily Close price** (`Close_t`).
A one-day lag in the target triggers the **Identity Function Trap** — the model learns "Tomorrow = Today."

### Strict Scaling Discipline
- Use `RobustScaler` or `MinMaxScaler`
- **Critical:** Scalers must be fit **strictly on the training partition**, then applied to validation/test sets
- Prevents data leakage

---

## V. ADVANCED FEATURE PHYSICS

### Primary Price Seeds

| Feature | Formula | Purpose |
|---|---|---|
| **HLC3** (Gold Standard) | `(High + Low + Close) / 3` | Low-pass filter — reduces wick volatility, retains price dynamics |
| **Medium Noise Filter** | `(High + Low) / 2` | Reduces extreme daily price spikes, enhances regime-shift generalization |

### Sovereign Pillars (Consensus Anchors)
- **TEMA** — Triple Exponential Moving Average
- **FWMA** — Fibonacci Weighted Moving Average
- **OBV** — On-Balance Volume
- **Bollinger Bands**

### Signal Hierarchy for DIRECTION (GRU)

| Lens | Role | Primary Indicators |
|---|---|---|
| **90-Day (Macro)** | The Anchor | COG, Shannon Entropy, ER, Hurst, Logistic Prob, VIDYA |
| **30-Day (Mid)** | Conviction | LinReg Z-Scores, Aroon, R-Squared (R²) |
| **10-Day (Micro)** | Kinetic | Dispersion Index (distance between SMA/TEMA/Kalman) |

### Interaction Engineering (Ratios)
Use engineered ratios to pre-digest non-linear relationships and mitigate multicollinearity:
- `HLC3 / SMA_5`
- `EMA_50 / EMA_200`

---

## VI. KINEMATIC PROFILING — THE MULTI-ORDER LENS

Represent each indicator seed through three orthogonal lenses:

| Lens | Order | Formula | Purpose |
|---|---|---|---|
| **Z-score** | Level 2 (Position) | Z-score over 90-day window | Centers signal, eliminates nominal bias |
| **_z_slope** | Level 3 (Velocity) | Linear slope of the Z-score | Detects trend momentum |
| **_z_sos** | Level 4 (Acceleration) | Slope of the Slope (SOS) | Captures momentum turning points before price turns |

> **The "Why":** Price crossing a level is a 1st-degree signal.
> We code for 2nd and 3rd-degree signals — e.g., feeding the model the
> Acceleration of the COG allows the GRU to detect trend exhaustion *before* price turns.

---

## VII. NORMALIZATION & LENSING LOGIC

| Indicator Type | Treatment | Rationale |
|---|---|---|
| **Cumulative pillars** (OBV, PVT) | WIN Lens — Windowed Percentage Change | Treats every window start as "0", reveals geometry not drift |
| **Oscillators & Ratios** (Entropy, RSI, Hurst) | Z-Lens — Z-score transformation | Centers on origin, satisfies scale-sensitivity |

---

## VIII. DIMENSIONALITY MANAGEMENT — THE PCA SQUEEZE

1. **Redundancy Filter:** Remove all features with Pearson correlation `|r| > 0.95`
2. **The 35-Component Rule:** Apply PCA, reduce to top 35 principal components targeting **95% total variance retention**
3. **Curation over Volume:** A curated subset of ~6 features (identified via judicial impact analysis) often matches high-dimensional raw data with lower computational latency

---

## IX. NEURAL ARCHITECTURE SPEC

### Hybrid CNN-LSTM Stack
```
Input → [1D-CNN with causal padding] → [LSTM/GRU] → Output
```
The CNN layer extracts spatial "shape" features (chart patterns) before sequential processing.

### Initialization & Stability
- **Glorot-Xavier initialization** for gradient stability

### Loss Functions

| Brain | Task | Loss Function |
|---|---|---|
| DIRECTION | Binary classification | Binary Cross-Entropy (BCE) |
| EASE | Regression | Huber Loss |
| EXPAND | Regression | Huber Loss |

---

## X. TRAINING & AUDIT PROTOCOLS

### Walk-Forward Loop
- **60-Day Rolling Lookback Window** with 1-day step size — mimics real-time trading dynamics
- **Epochs:** 100–300 with **Early Stopping patience of ~25 epochs**

### Judicial Audit Gating
- Terminate any training iteration if **Mean Impact (I_Norm) < 0.05** — indicates failure to find signal beyond noise

### Consensus Gating
- Combine brain outputs via **weighted integration**
- Trades only suggested when Direction, Ease, and Expansion **all align**

---

## XI. REPLACEMENT DECISIONS

| Removed | Replacement | Reason |
|---|---|---|
| **VPIN** | **R-Squared (R²)** | VPIN too noisy for 1-day direction; R² provides cleaner Conviction Score by measuring price-to-linear-trend fit |

---

## XII. TECH STACK

| Component | Technology |
|---|---|
| Backend | Python (cloud-native) |
| ML Framework | TensorFlow / Keras |
| Storage | h5py / HDF5 |
| Scaling | RobustScaler / MinMaxScaler (scikit-learn) |
| Dimensionality | PCA (scikit-learn) |

---

## FINAL DIRECTIVE

> *"Construct a cloud-native Python backend using TensorFlow/Keras and h5py for HDF5 storage.
> Implement a walk-forward optimization loop with 60-day rolling windows.
> Build the library using HLC3, TEMA, and OBV seeds processed through Triple-Order Lenses (Z, Slope, SOS).
> Submit the final tensor to a 35-component PCA squeeze for 95% variance retention.
> Use Huber Loss and Glorot-Xavier initialization to ensure gradient stability and reach the 0.014 MAE benchmark."*
