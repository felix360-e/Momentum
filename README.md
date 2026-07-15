# Cross-Sectional Momentum: ML Factor Research, Selection & Backtest

A full quantitative research pipeline — from raw OHLCV panel data to a validated, tradable factor — built to predict **21-day forward returns** across a US equity universe with **XGBoost**, and evaluated the way a systematic desk evaluates a signal: walk-forward, out-of-sample, with an honest information coefficient before a single trade is simulated.

This isn't a single script. It's an end-to-end system: feature engineering → statistical feature selection → cross-sectional ML → event-driven backtest → performance/factor diagnostics → a rule-based options overlay layer on top of the signal.

**Stack:** Python · pandas / Polars · XGBoost · scikit-learn · SciPy · Zipline (event-driven backtesting) · Alphalens (factor diagnostics) · Pyfolio (performance tear sheets) · ffn

---

## Results at a glance

| Metric | Value | What it means |
|---|---|---|
| Candidate features engineered | 84 | Technical, microstructure, momentum, and regime features from raw OHLCV |
| Reduced via correlation-cluster pruning | 84 → 30 | Hierarchical clustering on pairwise correlation, one representative per cluster |
| Final feature set (wrapper search) | 15 features | Forward/backward greedy selection optimizing walk-forward loss |
| Rank IC, 21-day horizon (final factor) | **0.0215**, t = 8.76, p ≈ 2.8e-18 | Statistically significant predictive power on forward returns |
| Rank IC, 63-day horizon | **0.0347**, t = 14.53, p ≈ 1.1e-46 | IC strengthens at longer horizons — signal decays slowly |
| Validation method | Walk-forward, embargoed train/test splits | No lookahead: test windows start after the forward-return horizon has fully elapsed |
| Backtest engine | Zipline (event-driven), monthly rebalance | Long-short factor portfolio, transaction costs and slippage modeled |

The 21D IC of ~0.02 is modest by design — that's what a real, tradable, non-overfit cross-sectional signal looks like. A "great" backtest IC (0.15+) on daily-return prediction is usually a lookahead bug, not alpha. I check for that explicitly (see [Validation methodology](#validation-methodology)).

---

## Table of contents

- [Problem](#problem)
- [Pipeline architecture](#pipeline-architecture)
- [1. Feature engineering](#1-feature-engineering)
- [2. Feature selection](#2-feature-selection)
- [3. Model & validation methodology](#3-model--validation-methodology)
- [4. Backtest (Zipline)](#4-backtest-zipline)
- [5. Factor diagnostics (Alphalens)](#5-factor-diagnostics-alphalens)
- [6. Options overlay decision engine](#6-options-overlay-decision-engine)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)

---

## Problem

Can a cross-sectional ML model, trained only on information available at decision time, predict which stocks in a liquid US universe will outperform over the next 21 trading days — after accounting for correlation-driven feature redundancy, transaction costs, and the risk of curve-fitting a walk-forward split?

The project is scoped as a **factor research study first, a backtest second** — the signal has to survive Spearman IC testing on held-out data before it ever gets converted into a portfolio.

## Pipeline architecture

```
Raw OHLCV panel (multi-symbol, daily)
        │
        ▼
Universe screener  →  liquidity / volatility / min-price filters, top-N by volume & momentum
        │
        ▼
Feature engineering →  84 candidate features (technical + microstructure + momentum + regime)
        │
        ▼
Feature selection   →  correlation-cluster pruning (84→30) → greedy forward/backward wrapper search (→15)
        │
        ▼
XGBoost regressor   →  predicts 21-day forward return, walk-forward validated, embargoed splits
        │
        ▼
Zipline backtest    →  monthly rebalance, long-short portfolio construction, costs + slippage
        │
        ▼
Diagnostics          →  Pyfolio tear sheet (returns/risk) + Alphalens (IC, quantiles, turnover)
```

## 1. Feature engineering

Beyond standard technical indicators (RSI, MACD, ADX, ATR/NATR, rolling stats), I built three custom feature families from raw OHLCV, each with an explicit economic rationale:

**Market microstructure & fragility**
- `amihud_illiquidity_21/63` — Amihud (2002) illiquidity: `|return| / dollar_volume`, rolled over 21/63 days
- `kyle_lambda_21` — a rolling price-impact coefficient (covariance of price change with dollar volume, normalized by volume variance) — a simplified Kyle's lambda
- `roll_spread_21` — Roll (1984) implied bid-ask spread from serial return covariance
- `fragility_score` — a cross-sectionally z-scored composite of illiquidity + price impact, flagging names where momentum is more likely to mean-revert sharply

**Normalized momentum & extension**
- `atr_extension_50/200` — distance from the 50/200-day SMA in ATR multiples (volatility-normalized, so it's comparable across regimes and price levels)
- `price_zscore_20/63` — rolling z-score of price, giving the model a stationary read on short-term exhaustion vs. medium-term trend

**Regime context**
- `kalman_momentum` / `kalman_velocity` — a constant-velocity Kalman filter on price, giving a phase-lag-free smoothed trend and its rate of change
- `ema_20_50_spread`, `ema_50_200_spread`, `ema_200_800_spread`, `ema_regime_score` — a four-timeframe EMA hierarchy compressed into continuous bull/bear regime scores, so the model can condition momentum predictions on the prevailing structure

All features are computed strictly on trailing data, grouped by symbol, with explicit `min_periods` guards — no cross-sectional or forward-looking leakage at the feature-construction stage.

## 2. Feature selection

84 candidate features is too many for a tree model trained on a few years of daily cross-sectional data — redundant, highly correlated features dilute importance and slow convergence without adding signal. I ran a four-stage reduction:


**Stage 1 — Gain-based importance.** An initial XGBoost fit ranks all 84 features by average loss improvement per split.
<img width="532" height="475" alt="Screenshot 2026-07-15 at 8 51 43 AM" src="https://github.com/user-attachments/assets/fdd9f7ce-0104-48e5-b7f7-e680d21d5b78" />

**Stage 2— Hierarchical correlation clustering.** Pairwise Pearson correlation is converted to a distance matrix (`1 − |corr|`), clustered with average-linkage hierarchical clustering, and cut at a distance threshold — collapsing 84 features into 30 clusters. Within each cluster, the feature with the highest correlation to the target survives.
<img width="757" height="370" alt="Screenshot 2026-07-15 at 8 45 01 AM" src="https://github.com/user-attachments/assets/f2577657-e839-4da8-a971-b9be79cb0684" />


**Stage 2 — Greedy forward/backward wrapper search.** Starting from the 30 cluster representatives, forward selection greedily adds the feature that most improves cross-validated loss; backward selection does the reverse from the full set. Both are walk-forward, not k-fold — folds respect time order so no future information leaks into feature selection itself.


**Stage 3 — Consolidation.** The surviving 15 features (technical indicators + the two microstructure/regime families above) plus 4 fundamental ratios (`debtToAssetsRatio`, `priceToBookRatio`, `incomeQuality`, `returnOnInvestedCapital`, `returnOnEquity`) from quarterly FMP data form the final predictor set, wired into a single `compute_selected_features()` function so the whole pipeline is reproducible from raw OHLCV with no manual steps.

## 3. Model & validation methodology

**Model:** `XGBRegressor` (`n_estimators=350, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7`, L1/L2 regularization) predicting 21-day forward return, evaluated with **Spearman rank IC** rather than raw R² — what matters for a cross-sectional strategy is whether the model ranks stocks correctly, not whether it fits the magnitude of the return.

**Why walk-forward, not k-fold:** financial time series aren't exchangeable — a k-fold split would let the model train on data that's chronologically after its test set, which is lookahead bias by another name. Every split here is a rolling train/test boundary in time order.

**Why embargoed splits:** the target is a 21-day *forward* return, so a naive split with the test window starting the day after training ends still leaks — the last `horizon` days of the training label window overlap the first days of the test period. Every split embargoes the boundary by the forward-return horizon before evaluating.

**Honesty check:** an early, deliberately unconstrained walk-forward run — fit on 84 raw features, only 6 years of data — produced a **negative held-out rank IC (−0.14)**. That's the expected failure mode of an under-regularized, over-featured model on a short window, and it's why the feature-selection and wider-history steps above exist. I'm including that number rather than hiding it — a pipeline that only shows you the number that worked isn't a validated pipeline.

## 4. Backtest (Zipline)

The final 15-feature + fundamentals model runs inside an event-driven **Zipline** backtest (2010–2026, monthly rebalance, realistic slippage and commission model) driving a long-short equal-weight portfolio (top 15 names by predicted signal, thresholded at ±1%).

<img width="1091" height="699" alt="Screenshot 2026-07-15 at 8 41 07 AM" src="https://github.com/user-attachments/assets/6f91c2b6-3250-4f39-aa77-212d772fa756" />


Full performance diagnostics via **Pyfolio** — cumulative and rolling returns, drawdown, rolling Sharpe, and position/transaction analytics — give the risk-manager-level view, not just a single equity curve:

<img width="281" height="560" alt="Screenshot 2026-07-15 at 8 42 03 AM" src="https://github.com/user-attachments/assets/c2ae37f8-d4b4-4333-978c-d61bc7514c22" />


## 5. Factor diagnostics (Alphalens)

Before trusting the backtest, the factor itself is diagnosed independently with **Alphalens**: clean factor/forward-return alignment, quantile bucketing, and IC computed at 5/10/21/63-day horizons. 
| Horizon | IC mean | t-stat | p-value |
|---|---|---|---|
| 5D | 0.0208 | 4.17 | 3.4e-05 |
| 10D | 0.0211 | 5.98 | 2.7e-09 |
| 21D | 0.0215 | 8.76 | 2.8e-18 |
| 63D | 0.0347 | 14.53 | 1.1e-46 |

<img width="543" height="206" alt="Screenshot 2026-07-15 at 8 53 20 AM" src="https://github.com/user-attachments/assets/83c046be-c252-436a-b09f-5a39385e9db3" />


## 6. Options overlay decision engine

A separate, deliberately simple module maps `(momentum signal, IV rank regime, portfolio objective)` to a specific options structure — e.g., bullish + high IV + income → bull put spread; bullish + low IV + growth → call debit spread. It's a transparent rule-based lookup table, not a model: every recommendation is traceable to the exact rule that produced it, and it's meant to sit downstream of the ML signal as a practitioner-facing translation layer, not to replace it.

## Tech stack

`Python` · `pandas` / `Polars` (dual-engine feature pipeline) · `NumPy` · `SciPy` (Spearman IC, t-tests) · `scikit-learn` · `XGBoost` · `Zipline` (event-driven backtesting) · `Alphalens` (factor diagnostics) · `Pyfolio` (performance tear sheets) · `ffn` · `yfinance` · `joblib` (parallel CV folds)

## Repository structure

```
.
├── README.md
├── images/                       # charts referenced above
├── notebooks/
│   └── ml_momentum_research.ipynb
├── features/
│   ├── microstructure.py         # Amihud, Kyle's lambda, Roll spread, fragility score
│   ├── momentum.py                # ATR extension, rolling z-scores
│   └── regime.py                  # Kalman filter, EMA hierarchy
├── selection/
│   └── feature_selection.py       # correlation-cluster pruning + forward/backward wrapper search
├── strategy/
│   └── config.py                  # backtest CONFIG: universe, features, model, portfolio construction
└── requirements.txt



