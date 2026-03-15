# SOVEREIGN TITAN v3.19 — Feature Intelligence Report

### Proprietary Signal Architecture & Indicator Taxonomy

**Classification:** Client-Facing Technical Documentation
**Version:** 3.19.18 — Entropy Injection Edition
**Prepared for:** Prospective Partners & Institutional Due Diligence

---

## Executive Summary

The Sovereign Titan system processes raw market data (Open, High, Low, Close, Volume) through a multi-layered feature engineering pipeline that produces **327 normalized signals** from **27 base indicators** and **3 unbounded deviation measures**. Each base indicator is subjected to a proprietary **Z-Lens normalization** across four temporal horizons (10, 30, 60, and 90 trading days), yielding three derivative transforms per horizon: the z-score itself, its slope (first derivative), and its slope-of-slope (second derivative — acceleration).

This document provides a comprehensive taxonomy of every indicator, its mathematical foundation, the information it encodes, and the structural relationships between signals. The system is designed so that no single indicator dominates — the Sovereign Hunt algorithm enforces diversity through correlation thresholds and family-level deduplication.

---

## Table of Contents

1. [Indicator Taxonomy Overview](#1-indicator-taxonomy-overview)
2. [Base Indicators — Detailed Profiles](#2-base-indicators--detailed-profiles)
   - 2.1 Moving Average Displacement Group
   - 2.2 Trend Efficiency & Momentum Group
   - 2.3 Statistical Structure Group
   - 2.4 Information Theory Group
   - 2.5 Breakout & Position Group
   - 2.6 Volatility & Dispersion Group
3. [Log-Ratio Protocol — Engineered Ratios](#3-log-ratio-protocol--engineered-ratios)
   - 3.1 Price Anchor Ratios
   - 3.2 Velocity Seed Ratios
   - 3.3 Interaction Ratios
4. [The Z-Lens Normalization Framework](#4-the-z-lens-normalization-framework)
5. [Feature Correlation & Uniqueness Matrix](#5-feature-correlation--uniqueness-matrix)
6. [Information Clustering Analysis](#6-information-clustering-analysis)
7. [Appendix A: Complete Feature Manifest](#appendix-a-complete-feature-manifest)
8. [Appendix B: Mathematical Definitions](#appendix-b-mathematical-definitions)

---

## 1. Indicator Taxonomy Overview

Every indicator in the system belongs to one of six functional families. Each family captures a fundamentally different dimension of market behavior:

| # | Family | What It Measures | Indicators | Unique Insight |
|---|--------|-----------------|------------|----------------|
| 1 | **Moving Average Displacement** | How far price has deviated from smoothed fair value | TEMA-30, SMA-20, HMA-21, Kalman Filter | Mean-reversion pressure; stretch vs. snap-back |
| 2 | **Trend Efficiency & Momentum** | Whether price movement is directed or noisy | ER-20, VIDYA CMO-20, ADX-14, Logistic Prob-30 | Distinguishes trending from choppy regimes |
| 3 | **Statistical Structure** | Whether price follows a predictable path | R²-30, Hurst-50, Linear Slope-30 | Trend persistence, linearity, and memory |
| 4 | **Information Theory** | How orderly or chaotic price distribution is | Shannon Entropy-20 | Regime detection through distributional complexity |
| 5 | **Breakout & Position** | Where price sits relative to its recent range | Donchian High-50, Aroon Up-25, COG-20 | Breakout proximity, range position, momentum timing |
| 6 | **Volatility & Dispersion** | How much smoothing methods disagree with each other | Dispersion-30, ATR-14 | Cross-model disagreement as a volatility proxy |

**Engineered ratios** (Section 3) combine indicators from different families to create cross-dimensional signals that no single indicator can produce alone.

---

## 2. Base Indicators — Detailed Profiles

### 2.1 Moving Average Displacement Group

These indicators measure **how far current price has stretched from a smoothed estimate of fair value**. They are the system's primary mean-reversion sensors.

---

#### TEMA-30 (Triple Exponential Moving Average, 30-period)

| Property | Value |
|----------|-------|
| **Classification** | Trend-Following Smoother |
| **Input** | HLC3 (average of High, Low, Close) |
| **Range** | Unbounded (pre-transformed to % deviation) |
| **Lookback** | 30 trading days |
| **Stationarity** | Non-stationary raw; stationary after hlc3/TEMA - 1 transform |

**What it measures:** TEMA applies three layers of exponential smoothing and then reconstructs a low-lag estimate: `TEMA = 3·EMA - 3·EMA² + EMA³`. This triple-pass architecture cancels most of the lag inherent in a simple EMA while preserving smoothness. The system feeds `hlc3 / TEMA - 1` (percent displacement) into the Z-Lens, not the raw TEMA value.

**Nuance:** TEMA is more responsive than SMA or EMA at the same period but can overshoot during whipsaw markets. Its displacement signal is strongest when price makes a clean directional move — the percent deviation grows monotonically. During choppy markets, TEMA displacement oscillates rapidly near zero, which is precisely the regime information the Z-Lens captures as low-z-score clustering.

**Institutional context:** TEMA is widely used in systematic trend-following as a signal smoother with minimal phase delay. Its inclusion here as a *displacement* measure (not a crossover signal) reflects a fundamentally different usage — measuring elastic stretch rather than directional bias.

---

#### SMA-20 (Simple Moving Average, 20-period)

| Property | Value |
|----------|-------|
| **Classification** | Trend-Following Smoother |
| **Input** | HLC3 |
| **Range** | Unbounded (pre-transformed to % deviation) |
| **Lookback** | 20 trading days |
| **Stationarity** | Stationary after hlc3/SMA - 1 transform |

**What it measures:** The arithmetic mean of the last 20 HLC3 values. Equal weighting of all observations means SMA responds slowly to new information but provides an unbiased estimate of recent central tendency.

**Nuance:** SMA-20 displacement (`hlc3/SMA - 1`) captures the "distance from consensus" — how far price has traveled from where the average market participant over the last month considers fair value. Because it weights all 20 bars equally, it is more robust to single-bar outliers than EMA-based measures but introduces ~10-bar lag. This lag is a *feature* in the system: SMA displacement signals persistence (the stretch must be sustained to grow) rather than reactivity.

**Differentiation from TEMA-30:** SMA-20 is shorter-horizon and higher-lag. When SMA-20 displacement and TEMA-30 displacement diverge, it signals that recent price action is accelerating or decelerating relative to the broader trend — a second-order insight the Z-Lens slope captures automatically.

---

#### HMA-21 (Hull Moving Average, 21-period)

| Property | Value |
|----------|-------|
| **Classification** | Adaptive Trend Smoother |
| **Input** | HLC3 |
| **Range** | Unbounded (pre-transformed to % deviation) |
| **Lookback** | 21 trading days (effective: ~10-bar response) |
| **Stationarity** | Stationary after hlc3/HMA - 1 transform |

**What it measures:** The Hull Moving Average uses weighted moving averages (WMA) to create a smoother that reduces lag to near-zero: `HMA = WMA(2·WMA(n/2) - WMA(n), sqrt(n))`. The system implements this via Numba-JIT weighted moving averages for computational efficiency.

**Nuance:** HMA is the most aggressive smoother in the system. It can actually *lead* price during strong trends because its differencing architecture (`2·WMA_short - WMA_long`) creates a forward-extrapolating component. This means HMA displacement can go negative *before* price reverses — a predictive rather than reactive signal. The trade-off is higher false-signal rate during ranging markets.

**Differentiation:** HMA provides the fastest displacement signal. When HMA displacement reverses before SMA or TEMA, it often signals the early stages of a trend exhaustion. The system captures this lead-lag relationship through the Z-Lens slope-of-slope (acceleration) transform.

---

#### Kalman Filter

| Property | Value |
|----------|-------|
| **Classification** | Adaptive Bayesian Smoother |
| **Input** | HLC3 |
| **Range** | Unbounded (pre-transformed to % deviation) |
| **Lookback** | Infinite (recursive; governed by R=0.0001, Q=0.001 parameters) |
| **Stationarity** | Stationary after hlc3/Kalman - 1 transform |

**What it measures:** A single-state Kalman filter that maintains a running estimate of "true price" by optimally balancing the measurement noise ratio (R) against the process noise ratio (Q). At each timestep, the filter computes a Kalman gain `K = P⁻/(P⁻ + R)` that determines how much weight to give new observations versus the prior estimate.

**Nuance:** Unlike fixed-window moving averages, the Kalman filter is *theoretically optimal* under Gaussian noise assumptions. With R=0.0001 and Q=0.001, the filter trusts its prior estimate heavily (Q/R = 10:1 ratio), making it extremely smooth during steady markets but adaptive during regime changes. The Kalman gain is high when uncertainty is high (after gaps or volatility spikes) and low during calm periods — an automatic regime-sensitivity that no fixed-window smoother can replicate.

**Institutional context:** Kalman filtering is the backbone of statistical arbitrage desks and GPS navigation systems alike. In this system, it serves as the "best estimate of fair value" against which all other smoothers are compared. The dispersion between Kalman and other MAs is itself an indicator (see Dispersion-30).

---

### 2.2 Trend Efficiency & Momentum Group

These indicators answer the question: **"Is the market trending, and if so, how efficiently?"**

---

#### ER-20 (Efficiency Ratio, 20-period)

| Property | Value |
|----------|-------|
| **Classification** | Trend Quality Measure |
| **Input** | HLC3 |
| **Range** | [0, 1] |
| **Lookback** | 20 trading days |
| **Stationarity** | Naturally stationary (bounded ratio) |

**Formula:** `ER = |Price_t - Price_{t-20}| / Σ|Price_t - Price_{t-1}| over 20 bars`

**What it measures:** The ratio of net displacement to total path traveled. An ER of 1.0 means price moved in a perfectly straight line over 20 days. An ER near 0.0 means price churned back and forth with no net progress.

**Nuance:** ER is Perry Kaufman's foundational contribution to adaptive trading. It distinguishes *trending* markets (high ER → wide stops, hold positions) from *mean-reverting* markets (low ER → tight stops, fade moves). Critically, ER is agnostic to direction — it measures the *quality* of a trend, not its sign. A strong downtrend and a strong uptrend produce identical ER values, making it a pure regime classifier.

**Institutional context:** ER is the backbone of Kaufman's Adaptive Moving Average (KAMA) and is used by major CTAs to dynamically adjust position sizing and stop distances. In this system, it serves as a direct input and as a denominator in several engineered ratios (ratio_breakout_eff, ratio_eff_slope).

---

#### VIDYA CMO-20 (Variable Index Dynamic Average — Chande Momentum Oscillator, 20-period)

| Property | Value |
|----------|-------|
| **Classification** | Directional Momentum Oscillator |
| **Input** | HLC3 |
| **Range** | [-1, +1] |
| **Lookback** | 20 trading days |
| **Stationarity** | Naturally stationary (bounded) |

**Formula:** `CMO = Σ(diffs over 20) / Σ(|diffs| over 20)`

**What it measures:** The signed version of the Efficiency Ratio. While ER measures trend quality without regard to direction, CMO preserves the sign: positive values indicate net upward momentum, negative values indicate net downward momentum. The magnitude indicates efficiency.

**Nuance:** CMO-20 is mathematically equivalent to `(up_sum - down_sum) / (up_sum + down_sum)`. It collapses when momentum is balanced (near zero) and extends to extremes during one-sided moves. Unlike RSI, which compresses into [0, 100] with a non-linear mapping, CMO's linear [-1, +1] range makes it naturally suited for z-score normalization and linear combination with other features.

**Relationship to ER-20:** `|CMO| = ER` when the net direction is aligned. CMO adds *directional information* that ER deliberately discards. The system uses both because a model benefits from knowing independently (a) how efficient the trend is and (b) which direction it favors.

---

#### ADX-14 (Average Directional Index, 14-period)

| Property | Value |
|----------|-------|
| **Classification** | Trend Strength Oscillator |
| **Input** | High, Low, Close |
| **Range** | [0, 100] (typically 10–60 in practice) |
| **Lookback** | 14 trading days (double-smoothed) |
| **Stationarity** | Naturally stationary (bounded) |

**What it measures:** ADX quantifies the *strength* of a trend regardless of direction. It is derived from the directional movement indicators (+DI and -DI), which separately measure upward and downward pressure, then combined as `ADX = 100 × SMA(|+DI - -DI| / (+DI + -DI))`.

**Nuance:** ADX is a second-derivative indicator — it measures the *rate of directional divergence* between buyers and sellers, then smooths it. This double-smoothing makes ADX extremely slow to react (it peaks well after a trend is established) but very reliable — false signals are rare. ADX below 20 is universally interpreted as "no trend"; above 40 is "strong trend." The system uses ADX as a regime gate: high ADX signals suggest the model should weight trend-following features more heavily.

**Institutional context:** ADX is a standard component of every institutional trend-following system. Its inclusion here is table-stakes, but its value comes from the Z-Lens treatment, which transforms the raw bounded oscillator into a regime-relative signal (is ADX high relative to its own recent history?).

---

#### Logistic Probability-30 (Sigmoid of Normalized Linear Regression Slope)

| Property | Value |
|----------|-------|
| **Classification** | Probabilistic Trend Direction |
| **Input** | HLC3 (via 30-bar linear regression slope) |
| **Range** | (0, 1) |
| **Lookback** | 30 trading days |
| **Stationarity** | Naturally stationary (bounded by sigmoid) |

**Formula:** `P = 1 / (1 + exp(-slope_30 / std(slope_30)))`

**What it measures:** The 30-bar OLS slope, normalized by its own standard deviation and passed through a logistic sigmoid. The output is a pseudo-probability of upward trend: P > 0.5 means the slope is positive (uptrend), P < 0.5 means negative (downtrend), and the distance from 0.5 reflects confidence.

**Nuance:** Raw linear regression slopes are in price units and are non-stationary across different assets. A $2/day slope means something very different for a $20 stock versus a $2,000 stock. The logistic transformation solves both problems simultaneously: (1) normalizing by the slope's own standard deviation makes it unit-free, and (2) the sigmoid bounds the output, preventing extreme slopes from dominating. The result is a clean probability-like signal that means the same thing across all assets.

**Institutional context:** This feature is conceptually similar to the trend-strength signals used in risk parity and momentum factor portfolios, where OLS slopes are normalized and ranked cross-sectionally.

---

### 2.3 Statistical Structure Group

These indicators probe the **mathematical character** of the price series — is it linear, persistent, or random?

---

#### R²-30 (Coefficient of Determination, 30-period)

| Property | Value |
|----------|-------|
| **Classification** | Trend Linearity Measure |
| **Input** | HLC3 |
| **Range** | [0, 1] |
| **Lookback** | 30 trading days |
| **Stationarity** | Naturally stationary (bounded) |

**What it measures:** The square of the Pearson correlation between the time index [0, 1, 2, ..., 29] and the HLC3 values over the last 30 bars. R² = 1.0 means price followed a perfect straight line. R² = 0.0 means there is zero linear relationship between time and price.

**Nuance:** R² is the most direct measure of *trend quality* in the system. While ER-20 measures efficiency of net displacement, R² measures how well a linear model fits the price path. A market can have high ER but low R² (e.g., a gap followed by flat consolidation creates net displacement without linearity). Conversely, high R² with low slope magnitude means price is moving linearly but slowly — a "quiet trend" that many momentum systems miss.

**Relationship to other indicators:** R² and ADX measure similar concepts (trend strength) from different mathematical foundations. R² uses correlation; ADX uses directional movement. Their agreement confirms a trend; their divergence signals a complex regime (e.g., trending with high volatility).

---

#### Hurst Exponent-50 (Hurst Estimate, 50-period)

| Property | Value |
|----------|-------|
| **Classification** | Fractal Persistence Measure |
| **Input** | HLC3 |
| **Range** | Approximately [0, 1] (0.5 = random walk) |
| **Lookback** | 50 trading days |
| **Stationarity** | Naturally stationary (bounded) |

**What it measures:** The Hurst exponent estimates the degree of long-range dependence in a time series. H > 0.5 indicates *persistence* (trends tend to continue), H < 0.5 indicates *anti-persistence* (movements tend to reverse), and H = 0.5 indicates a random walk.

**Nuance:** The system uses a simplified variance-ratio estimator: `H = log(std) / log(n)`. While this is less precise than rescaled range (R/S) analysis or DFA, it is computationally efficient for rolling calculation via Numba JIT. The 50-bar lookback provides a medium-term persistence estimate — long enough to detect genuine memory effects but short enough to adapt to regime changes.

**Critical insight:** Hurst is the only indicator in the system that directly addresses the *fractal structure* of returns. A declining Hurst from 0.7 to 0.5 signals that a previously persistent trend is losing its memory — often a precursor to reversal that pure momentum indicators miss entirely.

**Institutional context:** Hurst exponents are a cornerstone of fractal market hypothesis research (Edgar Peters, Benoit Mandelbrot). Their use in production trading systems is less common due to estimation noise, which is why the Z-Lens normalization is critical — it converts the noisy raw estimate into a regime-relative signal.

---

#### Linear Regression Slope-30 (lr_slope_30)

| Property | Value |
|----------|-------|
| **Classification** | Trend Velocity |
| **Input** | HLC3 |
| **Range** | Unbounded (price units per bar) |
| **Lookback** | 30 trading days |
| **Stationarity** | Non-stationary raw; stationary after Z-Lens |

**What it measures:** The OLS regression slope of HLC3 over 30 bars — the "speed" of the trend in price units per trading day. Positive slope = uptrend, negative = downtrend, magnitude = speed.

**Nuance:** This is the raw building block from which logistic_prob_30 is derived, but it is also passed directly into the Z-Lens as an independent feature. The reason: the Z-Lens transformation captures *how unusual the current slope is relative to its own recent history*, which is different information from the probabilistic direction signal. A slope that is positive but declining from a high z-score is a fundamentally different setup than a slope that is positive and rising.

---

### 2.4 Information Theory Group

---

#### Shannon Entropy-20

| Property | Value |
|----------|-------|
| **Classification** | Distributional Complexity Measure |
| **Input** | HLC3 |
| **Range** | [0, log(bins)] ≈ [0, 2.30] for 10 bins |
| **Lookback** | 20 trading days |
| **Stationarity** | Naturally stationary (bounded) |

**What it measures:** The Shannon entropy of the empirical distribution of HLC3 values over 20 bars, computed using a 10-bin histogram. High entropy = prices are spread uniformly across the range (high uncertainty, no dominant price level). Low entropy = prices cluster at one or two levels (low uncertainty, clear attractor).

**Nuance:** Entropy is the only indicator in the system that measures the *shape of the price distribution* rather than its central tendency or trend. A market with identical mean and standard deviation can have very different entropy depending on whether prices are uniformly distributed (high entropy) or bimodal (lower entropy). This makes entropy a unique regime detector: trending markets have low entropy (prices cluster at the trend front), while ranging markets have high entropy (prices visit many levels equally).

**Institutional context:** Information-theoretic measures are increasingly used in quantitative finance for regime detection and portfolio construction (e.g., entropy-based diversification). Shannon entropy is the most fundamental such measure and serves as a denominator in two engineered ratios (ratio_struct, adx_entropy_ratio), where it acts as a "randomness baseline" against which structure is measured.

---

### 2.5 Breakout & Position Group

---

#### Donchian High-50 (50-period Donchian Channel Position)

| Property | Value |
|----------|-------|
| **Classification** | Range Position / Breakout Proximity |
| **Input** | High prices |
| **Range** | [-1, 0] (0 = at the 50-day high) |
| **Lookback** | 50 trading days |
| **Stationarity** | Naturally stationary (bounded ratio) |

**Formula:** `donchian_high_50 = High / rolling_max(High, 50) - 1`

**What it measures:** How far below its 50-day high the current bar's high is, expressed as a percentage. A value of 0.0 means price is at a new 50-day high (breakout). A value of -0.15 means the high is 15% below its 50-day peak.

**Nuance:** This is a *proximity-to-breakout* measure. Richard Dennis's original Turtle Trading system used Donchian channel breakouts as its primary entry signal. This system's implementation is more nuanced: rather than a binary "breakout or not" trigger, the continuous percentage gives the model gradient information about *how close* price is to breaking out, enabling it to learn pre-breakout compression patterns.

---

#### Aroon Up-25 (25-period Aroon Oscillator, Up Component)

| Property | Value |
|----------|-------|
| **Classification** | Recency of High |
| **Input** | High prices |
| **Range** | [0, 1] |
| **Lookback** | 25 trading days |
| **Stationarity** | Naturally stationary (bounded) |

**Formula:** `aroon_up = argmax(High over 25 bars) / 25`

**What it measures:** How recently the highest high occurred within the 25-bar lookback, normalized to [0, 1]. A value of 1.0 means the highest high was the most recent bar (strong uptrend momentum). A value near 0.0 means the high occurred at the beginning of the window (momentum has faded).

**Nuance:** Aroon captures *temporal recency of extremes* — information that no moving average or slope can provide. It answers the question: "Is the trend making new highs now, or did it peak days ago?" This temporal dimension is particularly valuable for detecting trend exhaustion: when Aroon drops from 1.0 while ADX remains high, it signals that the trend is strong but no longer making fresh progress.

---

#### Center of Gravity-20 (COG)

| Property | Value |
|----------|-------|
| **Classification** | Momentum Timing Oscillator |
| **Input** | HLC3 |
| **Range** | Unbounded (price units; normalized via rolling %) |
| **Lookback** | 20 trading days |
| **Stationarity** | Non-stationary raw; stationary after WIN rolling % transform |

**Formula:** `COG = -Σ(i × price_i) / Σ(price_i)` for i in [1, n]

**What it measures:** The center of gravity of price over the lookback window — a weighted average where recent prices receive higher weight. Unlike standard weighted averages, COG is specifically designed to identify the *balance point* of the price series, analogous to the physical center of mass.

**Nuance:** COG is the only indicator processed through the **Rolling %** pipeline rather than the Z-Lens, because its raw values are in price units and genuinely unbounded. The `COG / rolling_mean(COG, N) - 1` transformation converts it into a mean-reverting oscillator that measures deviation from its own norm. COG leads price turns more reliably than moving average crossovers because it responds to the *distribution* of prices within the window, not just their average.

---

### 2.6 Volatility & Dispersion Group

---

#### Dispersion-30 (Moving Average Disagreement)

| Property | Value |
|----------|-------|
| **Classification** | Cross-Model Volatility Proxy |
| **Input** | SMA-30, TEMA-30, Kalman Filter |
| **Range** | [0, ∞) (typically 0.001–0.05) |
| **Lookback** | 30 trading days (inherited from constituent MAs) |
| **Stationarity** | Approximately stationary (variance of ratios) |

**Formula:** `dispersion = std([hlc3/SMA-1, hlc3/TEMA-1, hlc3/Kalman-1])`

**What it measures:** The standard deviation of displacement ratios across three different smoothing methods. When all three MAs agree on fair value, dispersion is near zero. When they disagree significantly, dispersion is elevated.

**Nuance:** This is a second-order volatility measure that captures *model uncertainty* rather than price volatility. ATR measures how much price moves; dispersion measures how much different estimation methods disagree about where price *should* be. High dispersion often precedes regime changes because different smoothers respond at different speeds — during trend transitions, fast smoothers (Kalman, TEMA) lead while slow smoothers (SMA) lag, creating maximum disagreement.

**Institutional context:** This concept is analogous to the "model spread" used in options market-making, where the disagreement between pricing models (Black-Scholes, local vol, stochastic vol) is itself informative about mispricing opportunities.

---

#### ATR-14 (Average True Range, 14-period)

| Property | Value |
|----------|-------|
| **Classification** | Volatility Measure |
| **Input** | High, Low, Close |
| **Range** | [0, ∞) (price units) |
| **Lookback** | 14 trading days |
| **Stationarity** | Non-stationary (price-level dependent) |

**What it measures:** The rolling average of the "true range" — the maximum of (High-Low, |High-PrevClose|, |Low-PrevClose|) — over 14 bars. ATR captures the average daily price movement including gaps.

**Nuance:** ATR is not directly input to the Z-Lens (it is non-stationary in price units). Instead, it serves as a building block for the **ratio_snr** engineered ratio, where it acts as a noise denominator: `slope / ATR` measures how much signal (directional movement) exists relative to noise (volatility). This signal-to-noise concept is foundational in telecommunications engineering and translates directly to trend quality assessment.

---

## 3. Log-Ratio Protocol — Engineered Ratios

The Log-Ratio Protocol is the system's proprietary feature engineering layer. Raw ratios between indicators suffer from three problems: non-stationarity, scale sensitivity, and heavy-tailed distributions. The protocol applies mathematically rigorous log-transformations to produce **scale-invariant, distribution-stabilized** cross-dimensional signals.

### Transformation Rules

| Ratio Category | Raw Formula | Log-Ratio Protocol Transform |
|----------------|-------------|------------------------------|
| **Price Anchors** | A / B | log(A) − log(B) |
| **Velocity Seeds** | Slope_A / Slope_B | sign(ratio) × log(\|ratio\|) |
| **Interactions** | Slope × Indicator | sign(slope) × log(\|slope × indicator\|) |

**Why logarithms?** Log-transforms convert multiplicative relationships into additive ones, stabilize variance across price levels, and prevent high-priced assets from dominating the feature space. A 2% move on a $500 stock and a $50 stock produce the same log-ratio value.

---

### 3.1 Price Anchor Ratios

These ratios compare two smoothing methods to measure **relative displacement** — which smoother sees price as more stretched.

---

#### kalman_sma_ratio — Adaptive vs. Static Fair Value

| Property | Value |
|----------|-------|
| **Formula** | log(\|Kalman\|) − log(\|SMA-20\|) |
| **Category** | Price Anchor |
| **Measures** | Agreement between adaptive and fixed smoothers |
| **Range** | Unbounded (approximately [-0.01, +0.01] for equities) |

**Insight:** When the Kalman filter and SMA-20 agree (ratio near zero), the market is in a stable regime. When they diverge, the Kalman filter has detected information that the fixed SMA has not yet incorporated. Positive values mean Kalman is above SMA (Kalman sees higher fair value); negative means Kalman is below. The log-ratio ensures this comparison is meaningful across all price levels.

---

#### tema_kalman_ratio — Fast vs. Adaptive Fair Value

| Property | Value |
|----------|-------|
| **Formula** | log(\|TEMA-30\|) − log(\|Kalman\|) |
| **Category** | Price Anchor |
| **Measures** | Speed disagreement between smoothers |

**Insight:** TEMA responds faster than Kalman due to its triple-EMA construction. When TEMA leads Kalman (positive ratio), a new trend is accelerating. When Kalman leads TEMA (negative ratio), the adaptive filter has information that the fixed-lag smoother misses — often a sign of a developing reversal.

---

#### exhaustion_60 — Distance from 60-Day Peak

| Property | Value |
|----------|-------|
| **Formula** | log(Close) − log(rolling_max(High, 60)) |
| **Category** | Price Anchor |
| **Measures** | How far below the 60-day high the market has fallen |
| **Range** | (-∞, 0] (0 = at the peak) |

**Insight:** A scale-invariant exhaustion measure. A value of -0.05 means price is approximately 5% below its 60-day high, regardless of whether the stock trades at $20 or $2,000. The deeper the negative value, the more "exhausted" the uptrend. This log-ratio version avoids the bias that simple division creates for high-priced assets.

---

### 3.2 Velocity Seed Ratios

These ratios compare **rates of change** across different time horizons or normalization bases.

---

#### ratio_acc — Short vs. Medium-Term Acceleration

| Property | Value |
|----------|-------|
| **Formula** | sign(slope₁₀/slope₂₀) × log(\|slope₁₀/slope₂₀\|) |
| **Category** | Velocity Seed |
| **Measures** | Whether the trend is accelerating or decelerating |

**Insight:** When the 10-bar slope exceeds the 20-bar slope (ratio > 1, log > 0), the trend is accelerating — recent momentum exceeds the broader trend. When the ratio inverts, the trend is decelerating. The sign-preserving log transform handles the fact that slopes can be negative (downtrends) while preserving the acceleration/deceleration information.

---

#### ratio_snr — Trend Signal-to-Noise

| Property | Value |
|----------|-------|
| **Formula** | sign(slope₂₀/ATR₁₄) × log(\|slope₂₀/ATR₁₄\|) |
| **Category** | Velocity Seed |
| **Measures** | How much directional signal exists relative to volatility noise |

**Insight:** Borrowed from signal processing theory. A high SNR means the trend is strong relative to the market's random fluctuations — trades in the trend direction have favorable expected outcomes. A low SNR means noise dominates signal — trend-following strategies are likely to whipsaw.

---

#### curvature_diff — Trend Curvature (Short vs. Long Slope)

| Property | Value |
|----------|-------|
| **Formula** | sign(slope₁₀ − slope₆₀) × log(\|slope₁₀ − slope₆₀\|) |
| **Category** | Velocity Seed |
| **Measures** | Rate of change of the rate of change (second derivative) |

**Insight:** This is the discrete analog of mathematical curvature. Positive curvature means the price curve is bending upward (accelerating uptrend or decelerating downtrend). Negative curvature means the curve is bending downward. The 10 vs. 60 bar differential captures medium-term curvature rather than short-term noise.

---

#### cycle_vs_trend — Cyclicality Relative to Trend Slope

| Property | Value |
|----------|-------|
| **Formula** | sign(dominant_cycle₂₀/slope₃₀) × log(\|dominant_cycle₂₀/slope₃₀\|) |
| **Category** | Velocity Seed |
| **Measures** | Whether market cyclicality is high or low relative to trend strength |

**Insight:** The dominant cycle is estimated via zero-crossing frequency of the first derivative — a simple but robust method for detecting the prevailing oscillation period (clamped to 5–60 bars). When this cycle period is long relative to the trend slope, the market is in a slow, trending mode. When the cycle period is short relative to slope, the market is oscillating rapidly — a regime where mean-reversion strategies outperform trend-following.

---

### 3.3 Interaction Ratios

These ratios multiply or divide indicators from different families to create **cross-dimensional** signals.

---

#### ratio_eff_slope — Trend Efficiency × Velocity

| Property | Value |
|----------|-------|
| **Formula** | sign(slope₂₀) × log(\|slope₂₀ × ER₂₀\|) |
| **Category** | Interaction |
| **Measures** | Directional velocity weighted by trend quality |

**Insight:** Raw slope tells you how fast price is moving. ER tells you how efficient that movement is. Their product is a "conviction-weighted velocity" — a fast trend with high efficiency is far more meaningful than a fast trend with low efficiency (which is likely a whipsaw). The log-transform prevents extreme slope values during high-volatility regimes from dominating.

---

#### ratio_pers_slope — Trend Persistence × Velocity

| Property | Value |
|----------|-------|
| **Formula** | sign(slope₂₀) × log(\|slope₂₀ × Hurst₅₀\|) |
| **Category** | Interaction |
| **Measures** | Directional velocity weighted by fractal persistence |

**Insight:** Similar to ratio_eff_slope but weights by Hurst exponent instead of efficiency ratio. High Hurst (persistent) × strong slope = a trend with long-range memory that is statistically likely to continue. Low Hurst × strong slope = a trend that is moving fast but lacks persistence — a potential mean-reversion candidate.

---

#### ratio_struct — Linear Structure vs. Randomness

| Property | Value |
|----------|-------|
| **Formula** | log(\|R²₂₀\|) − log(Shannon₂₀) |
| **Category** | Semi-bounded Log-Ratio |
| **Measures** | How much linear structure exists relative to distributional randomness |

**Insight:** R² measures linearity; Shannon entropy measures randomness. Their log-ratio quantifies the market's position on the order-chaos spectrum. High ratio_struct = highly ordered, linear market. Low ratio_struct = chaotic, unpredictable distribution. This is the system's primary *regime classification* signal.

---

#### adx_entropy_ratio — Trend Strength vs. Randomness

| Property | Value |
|----------|-------|
| **Formula** | log(\|ADX₁₄\|) − log(Shannon₂₀) |
| **Category** | Semi-bounded Log-Ratio |
| **Measures** | Directional trend strength relative to distributional chaos |

**Insight:** Similar to ratio_struct but uses ADX (a trend strength measure derived from directional movement) instead of R² (a linear fit measure). When ADX is high and entropy is low, the market is in a clean trend with an ordered distribution — the highest-conviction setup. When ADX is low and entropy is high, the market is noisy and directionless.

---

#### ratio_breakout_eff — Breakout Proximity vs. Trend Efficiency

| Property | Value |
|----------|-------|
| **Formula** | log(\|Donchian_High_20\|) − log(ER₂₀) |
| **Category** | Semi-bounded Log-Ratio |
| **Measures** | Whether a breakout is occurring in an efficient or noisy context |

**Insight:** A breakout (Donchian near zero) in a high-efficiency environment (high ER) is fundamentally different from a breakout in a noisy environment. This ratio distinguishes "real" breakouts backed by clean directional movement from "false" breakouts driven by random volatility spikes.

---

## 4. The Z-Lens Normalization Framework

Every base indicator and every log-ratio passes through the Z-Lens, which produces **three derivative features per temporal horizon**:

### Transform Definitions

| Transform | Formula | What It Captures |
|-----------|---------|-----------------|
| **z** | `(value - rolling_mean) / rolling_std` | How extreme the current value is relative to recent history |
| **z_slope** | `OLS_slope(z, window)` | The *rate of change* of the z-score — is the indicator becoming more or less extreme? |
| **z_sos** | `OLS_slope(z_slope, window)` | The *acceleration* of the z-score — is the rate of change itself changing? |

### Temporal Horizons

| Horizon | Window | Captures |
|---------|--------|----------|
| **LENS_10** | 10 bars | Intraweek regime shifts; fast-twitch signals |
| **LENS_30** | 30 bars | Monthly regime context; medium-term normalization |
| **LENS_60** | 60 bars | Quarterly cycle context; institutional rebalancing windows |
| **LENS_90** | 90 bars | Seasonal context; long-term regime baseline |

### Feature Explosion

Each of 27 base indicators × 4 horizons × 3 transforms = **324 LENS features**, plus 3 COG rolling-% features = **327 total**.

The Z-Lens is critical for two reasons:
1. **Cross-asset comparability:** A z-score of +2.0 means the same thing for Apple and for crude oil — "the indicator is 2 standard deviations above its recent mean."
2. **Regime detection:** The z-score measures deviation from the *local* norm, so a feature value that is "normal" in a trending market can register as "extreme" in a ranging market, and vice versa.

---

## 5. Feature Correlation & Uniqueness Matrix

The following matrix shows expected correlation clusters among the 27 base indicators. Features within the same cluster provide *redundant* information; features across clusters provide *complementary* information. The Sovereign Hunt algorithm enforces a 0.85 correlation cap to prevent any cluster from dominating.

### High-Correlation Clusters (Expected r > 0.60)

| Cluster | Members | Shared Information |
|---------|---------|-------------------|
| **A: MA Displacement** | tema_30_pct, sma_20_pct, hma_21_pct, kalman_pct | All measure hlc3/MA - 1; differ only in lag and smoothness |
| **B: Trend Strength** | er_20, adx_14, r_sq_30 | All measure "is the market trending?" via different methods |
| **C: Directional Velocity** | lr_slope_30, logistic_prob_30, vidya_cmo_20 | All encode trend direction and speed |
| **D: MA Comparison** | kalman_sma_ratio, tema_kalman_ratio | Both compare smoothers; differ in which pair |
| **E: Slope Interactions** | ratio_eff_slope, ratio_pers_slope | Both weight slope by a quality measure |
| **F: Structure vs. Chaos** | ratio_struct, adx_entropy_ratio | Both compare order to randomness |

### Low-Correlation Pairs (Expected r < 0.25) — Maximum Uniqueness

| Feature A | Feature B | Why They're Uncorrelated |
|-----------|-----------|-------------------------|
| shannon_20 | aroon_up_25 | Distribution shape vs. temporal recency of highs |
| hurst_50 | dispersion_30 | Fractal persistence vs. cross-model disagreement |
| cog_20 | exhaustion_60 | Momentum timing vs. distance from peak |
| ratio_snr | donchian_high_50 | Signal-to-noise quality vs. range position |
| curvature_diff | ratio_struct | Second-derivative geometry vs. order/chaos regime |
| cycle_vs_trend | ratio_acc | Cyclicality period vs. slope acceleration |

### Information Orthogonality Summary

| Dimension | Primary Indicator(s) | Information Unique To This Dimension |
|-----------|---------------------|--------------------------------------|
| **Mean-reversion pressure** | MA displacements (A) | "How stretched is price?" |
| **Trend quality** | ER, ADX, R² (B) | "Is the movement efficient?" |
| **Trend direction** | CMO, Slope, Logistic (C) | "Which way?" |
| **Fractal memory** | Hurst-50 | "Will this persist?" |
| **Distributional order** | Shannon-20 | "How chaotic is the distribution?" |
| **Breakout proximity** | Donchian, Aroon (E) | "Where in the range?" |
| **Model disagreement** | Dispersion-30 | "Do smoothers agree?" |
| **Momentum timing** | COG-20 | "When is the center of mass?" |
| **Cross-dimensional** | All engineered ratios | "How do dimensions interact?" |

---

## 6. Information Clustering Analysis

The system's 27 base indicators can be organized into an information hierarchy with three tiers:

### Tier 1: Independent Signal Sources (Low mutual correlation)

These indicators provide genuinely orthogonal information. Each one captures a market dimension that no other indicator can replicate:

| # | Indicator | Unique Dimension |
|---|-----------|-----------------|
| 1 | **Hurst-50** | Fractal memory / persistence vs. anti-persistence |
| 2 | **Shannon Entropy-20** | Distributional complexity / regime order |
| 3 | **Dispersion-30** | Cross-model agreement / estimation uncertainty |
| 4 | **COG-20** | Volume-weighted momentum timing |
| 5 | **Aroon Up-25** | Temporal recency of price extremes |
| 6 | **exhaustion_60** | Scale-invariant distance from recent peak |
| 7 | **cycle_vs_trend** | Oscillation period relative to trend strength |

### Tier 2: Semi-Redundant Signal Families (Moderate internal correlation)

These indicators measure similar concepts via different mathematical methods. The Sovereign Hunt algorithm selects at most one or two from each family:

| Family | Members | Common Thread |
|--------|---------|---------------|
| **Trend Strength** | ER-20, ADX-14, R²-30 | "Is the market trending?" (correlation, directional movement, linearity) |
| **Trend Direction** | CMO-20, lr_slope_30, logistic_prob_30 | "Which way is the trend?" (signed efficiency, OLS slope, sigmoid probability) |
| **Fair Value Stretch** | tema_30_pct, sma_20_pct, hma_21_pct, kalman_pct | "How far from fair value?" (triple EMA, simple avg, Hull, Bayesian) |

### Tier 3: Engineered Combinations (Derived orthogonality)

These are the log-ratio features that combine Tier 1 and Tier 2 indicators to create cross-dimensional signals:

| Ratio | Tier 1 Source | Tier 2 Source | What It Creates |
|-------|---------------|---------------|-----------------|
| ratio_struct | Shannon-20 | R²-30 | Order-chaos spectrum |
| ratio_pers_slope | Hurst-50 | slope_20 | Memory-weighted velocity |
| ratio_snr | ATR-14 | slope_20 | Signal-to-noise quality |
| ratio_breakout_eff | Donchian-20 | ER-20 | Breakout conviction |
| ratio_acc | slope_10 | slope_20 | Trend acceleration |
| curvature_diff | slope_10 | slope_60 | Second-order geometry |

---

## Appendix A: Complete Feature Manifest

### Total Feature Count: 327

| Component | Count | Calculation |
|-----------|-------|-------------|
| Z-LENS base indicators | 15 | (11 bounded + 4 MA displacements) |
| Z-LENS log-ratio features | 12 | (3 price anchors + 4 velocity seeds + 5 interactions) |
| Z-LENS sub-features | 324 | 27 indicators × 4 horizons × 3 transforms |
| Rolling % features (COG) | 3 | 1 indicator × 3 windows |
| **Grand Total** | **327** | |

### Full Indicator → Feature Mapping

| Base Indicator | LENS_10 | LENS_30 | LENS_60 | LENS_90 | Total |
|----------------|---------|---------|---------|---------|-------|
| er_20 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| vidya_cmo_20 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| r_sq_30 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| hurst_50 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| shannon_20 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| adx_14 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| logistic_prob_30 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| aroon_up_25 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| donchian_high_50 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| dispersion_30 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| lr_slope_30 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| tema_30_pct | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| sma_20_pct | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| hma_21_pct | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| kalman_pct | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| kalman_sma_ratio | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| tema_kalman_ratio | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| exhaustion_60 | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| ratio_acc | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| ratio_snr | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| curvature_diff | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| cycle_vs_trend | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| ratio_eff_slope | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| ratio_pers_slope | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| ratio_struct | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| adx_entropy_ratio | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| ratio_breakout_eff | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | z, z_slope, z_sos | 12 |
| cog_20 (Rolling %) | WIN_10 | WIN_30 | WIN_60 | — | 3 |
| | | | | **TOTAL** | **327** |

---

## Appendix B: Mathematical Definitions

### Notation

- `hlc3` = (High + Low + Close) / 3
- `cl` = Close price
- `hi` = High price
- `lo` = Low price
- `vol` = Volume
- `Σ` = sum over the rolling window
- `n` = window length

### Core Formulas

**Efficiency Ratio (ER):**
```
ER(n) = |hlc3_t - hlc3_{t-n}| / Σ_{i=1}^{n} |hlc3_i - hlc3_{i-1}|
```

**VIDYA CMO:**
```
CMO(n) = Σ_{i=1}^{n} (hlc3_i - hlc3_{i-1}) / Σ_{i=1}^{n} |hlc3_i - hlc3_{i-1}|
```

**Hull Moving Average:**
```
HMA(n) = WMA(2 × WMA(n/2) - WMA(n), √n)
```

**Kalman Filter (single-state):**
```
P⁻_t = P_{t-1} + Q
K_t = P⁻_t / (P⁻_t + R)
x̂_t = x̂_{t-1} + K_t × (price_t - x̂_{t-1})
P_t = (1 - K_t) × P⁻_t
```

**Shannon Entropy (discrete):**
```
H = -Σ_{i=1}^{bins} p_i × log(p_i)
```

**Hurst Exponent (variance-ratio estimate):**
```
H ≈ log(std(series)) / log(n)
```

**R² (coefficient of determination):**
```
R² = [Σ(i - ī)(y_i - ȳ)]² / [Σ(i - ī)² × Σ(y_i - ȳ)²]
```

**Z-Lens Transform:**
```
z = (value - μ_rolling) / (σ_rolling + ε)
z_slope = OLS_slope(z, window)
z_sos = OLS_slope(z_slope, window)
```

**Log-Ratio Protocol:**
```
Price Anchors:   log(|A| + ε) - log(|B| + ε)
Velocity Seeds:  sign(A/B) × log(|A/B| + ε)
Interactions:    sign(slope) × log(|slope × indicator| + ε)
```

---

*This document is proprietary to the Sovereign Titan research program. Feature selection, normalization architecture, and ratio engineering represent original intellectual property developed through iterative empirical testing across 200+ liquid equities and ETFs.*
