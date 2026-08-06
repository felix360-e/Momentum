# Dynamic-Universe Technical Factor Selection & Backtest

A cross-sectional equity factor pipeline that builds a point-in-time-correct
liquid universe from raw OHLCV, screens ~100 engineered technical/
microstructure/momentum/regime features down to a small non-redundant set
using five independent filters, and backtests the survivors as an XGBoost
ranking signal in a long-only monthly-rebalanced portfolio.

## Problem

Most factor-selection workflows either eyeball a correlation heatmap and
call it a day, or keep every feature that clears a single IC threshold and
call the result "selected." Both approaches are vulnerable to noise: a
feature can show a genuine-looking IC on the full sample and still have a
sign that flips depending on which months you sample, or a feature can rank
#1 by one importance method and #8 by another with no real agreement
between them.

This project asks: **starting from ~100 technical, microstructure,
momentum, and regime features computed on a dynamic, point-in-time-correct
equity universe, which features survive redundancy removal, multiple-testing
correction, and out-of-sample stability checks — and does what survives
actually produce a tradable signal?**

## Data & universe construction

- **Source**: bulk daily OHLCV pulled from an FMP-backed data warehouse.
  Raw panel: **24,774,758 rows, 12,332 symbols, 2000-01-03 to 2026-07-10**.
- **Eligibility filters** (evaluated causally, no lookahead, on raw
  close/volume only — computed *before* any indicator touches the data, to
  avoid wasting compute on symbols that will fail the liquidity bar anyway):
  - Price floor: close ≥ $15
  - 60-day rolling average dollar volume ≥ $3,000,000
  - Minimum trading history: 1,260 days (5 years) before a symbol becomes
    eligible
  - Per-date liquidity cap: top 500 most liquid names only
  - A symbol needs at least 60 eligible dates somewhere in its history to be
    kept at all
- **Funnel**: 933,744 / 24,774,758 rows (3.8%) pass the eligibility test on
  any given date → 1,002 / 12,332 symbols are eligible at least once → 756
  symbols clear the `MIN_ELIGIBLE_DATES` bar and enter the reduced panel
  (3,296,152 rows).
- **Why reduce at the symbol level, not the row level**: indicator functions
  need continuous lookback history to compute rolling windows correctly:
  dropping a stock's pre-eligible days would break something like a 200-day
  moving average for its first ~200 days post-eligibility. So the *symbol's
  full history* is kept once it clears the bar on any date; the *row-level*
  `eligible` flag is what actually restricts the IC/backtest universe later.
- **Target**: 21-trading-day forward return, computed after indicators
  (`Return_fwd_21`).
- **Rebalance-date universe**: 258 monthly rebalance dates, 2005-01-31 to
  2026-06-09. Eligible universe size per month — min 1, median 230, max 297
  (the min-of-1 is an early-history artifact worth checking before trusting
  the earliest few rebalances).

## Method

The pipeline runs as an ordered funnel, each stage cheaper to compute than
the last because it operates on fewer symbols/features:

1. **Filter tickers on raw OHLCV first** (12,332 → 756 symbols) — before any
   indicator is computed, since eligibility only needs close/volume/date.
2. **Compute 101 technical, microstructure, momentum, and regime features**
   on the reduced 756-symbol panel (MACD variants, ROC, RSI, ADX, Hurst
   exponent, EWMA volatility, drawdown, z-scores, Amihud illiquidity, Kyle's
   lambda, roll spread, and more).
3. **Cross-sectional Information Coefficient (IC)**: Spearman rank
   correlation between each feature and the 21-day forward return, computed
   per rebalance date, then aggregated with a Newey-West HAC t-stat
   (21 lags, matched to the return horizon's overlap). 13 features were
   excluded outright for having no cross-sectional variation on a given date
   (e.g. `adjOpen`, several rolling-risk-metric columns).
4. **Correlation filtering**: features pairwise correlated above |0.90| are
   grouped and only the highest-|IC| representative from each group is kept
   — 88 features → 49.
5. **Hierarchical clustering** (complete linkage, `1 − |correlation|`
   distance) groups the 49 survivors into 10 clusters; one representative
   per cluster is kept by highest |IC| → 10 representatives.
6. **Benjamini-Hochberg FDR correction** on the Newey-West p-values from step
   3, run across all 88 originally-tested features: 4 were significant at
   raw p<0.05, but **0 survived BH-FDR correction**. This is reported
   honestly rather than discarded — it's a real signal that naive
   significance testing on this feature set overstates what's actually
   there.
7. **IC threshold filter**: representatives with |IC| ≥ 0.01, top 10 by
   |IC| → **8 final features**.
8. **Block-bootstrap stability selection**: 50 resamples of whole rebalance
   dates (not individual rows, to preserve cross-sectional structure and
   avoid overstating stability for slow-moving features) — see Results for
   what this revealed.
9. **ML importance cross-check**: XGBoost MDI (impurity-based) vs. sklearn
   permutation importance, compared against the IC ranking.
10. **Post-selection verification**: residual correlation matrix on the
    final feature set to confirm redundancy is actually gone.

## Results

### Selection funnel

| Stage | Features remaining |
|---|---|
| Initial engineered features | 101 (88 after dropping 13 with no cross-sectional variation) |
| After correlation filter (\|r\| ≥ 0.90) | 49 |
| Cluster representatives (10 clusters) | 10 |
| After IC filter (\|IC\| ≥ 0.01, top 10) | 8 |
| **Final selected** | **8** (92.1% removal rate) |

### Final 8 features (point-estimate IC, full sample)

| # | Feature | IC | What it captures |
|---|---|---|---|
| 1 | `close_drawdown` | +0.0312 | Distance below rolling peak |
| 2 | `atr_extension_200` | +0.0301 | Price extension relative to a 200-day ATR band |
| 3 | `close_fastqsmom_21_252_126` | +0.0272 | Composite fast momentum |
| 4 | `close_macd_line_50_200_30` | +0.0255 | Slow MACD line (50/200/30) |
| 5 | `close_roc_0_21` | −0.0183 | 21-day rate of change |
| 6 | `close_adx_63` | +0.0177 | 63-day trend strength (ADX) |
| 7 | `close_obv` | +0.0173 | On-balance volume |
| 8 | `close_roc_0_5` | −0.0124 | 5-day rate of change |

### The stability check changes the picture

This is the most important result in the notebook. Block-bootstrap
resampling (50 draws, by date) shows that **half of the 8 features that
passed the point-estimate IC screen have a sign that is not stable
out-of-sample**:

| Feature | Bootstrap IC mean | IC/IR | % of resamples with positive IC |
|---|---|---|---|
| `close_adx_63` | 0.0148 | **5.23** | 100% |
| `close_fastqsmom_21_252_126` | 0.0148 | **2.50** | 100% |
| `close_drawdown` | 0.0051 | 1.09 | 88% |
| `close_obv` | 0.0027 | 0.87 | 86% |
| `close_macd_line_50_200_30` | −0.0087 | **−1.58** | 8% |
| `atr_extension_200` | −0.0215 | **−3.37** | 0% |
| `close_roc_0_5` | −0.0258 | **−3.98** | 0% |
| `close_roc_0_21` | −0.0382 | **−5.24** | 0% |

Only `close_adx_63` and `close_fastqsmom_21_252_126` are both strong *and*
stable. `atr_extension_200`, `close_roc_0_5`, and `close_roc_0_21` had a
*positive* full-sample IC but a *negative* mean IC under resampling with
0% of bootstrap draws agreeing on the sign — a textbook case of a feature
that looks good on a point estimate and falls apart under out-of-sample
scrutiny. Anyone extending this pipeline should treat those three as
demoted, not confirmed.

### ML importance doesn't agree with itself

XGBoost MDI (impurity) and permutation importance were computed on the
final 8 features. Both methods surfaced the same top features by rank, but
**the Spearman correlation between the two importance rankings was 0.000**
— no agreement on relative ordering. This is reported as-is rather than
picking whichever method told a nicer story.

### Post-selection redundancy check

Maximum pairwise correlation among the final 8 features: **0.481** — well
below the 0.90 threshold used to filter earlier, confirming the redundancy
removal worked.

### Backtest (XGBoost ranking signal, long-only, monthly rebalance, 10 equal-weighted positions)

The 8 selected features were fed into an XGBoost regressor, retrained each
month to predict forward 21-day returns, and used to rank the eligible
universe. The top 10 names were held equal-weighted (10% each) and
rebalanced monthly via Zipline, benchmarked against SPY.

| Metric | Value |
|---|---|
| Backtest period | 2010-01-04 to 2026-05-08 (195 months) |
| Annual return | 22.18% |
| Cumulative return | 2,531.4% |
| Annual volatility | 29.18% |
| **Sharpe ratio** | **0.83** |
| Sortino ratio | 1.17 |
| Calmar ratio | 0.48 |
| **Max drawdown** | **−46.52%** (Feb–Mar 2020, recovered by Aug 2020) |
| Alpha / Beta (vs. SPY) | 0.08 / 1.28 |
| Daily turnover | 6.34% |
| Skew / Kurtosis | −0.43 / 5.50 |

Beta of 1.28 and a second, much longer drawdown (44.9%, Feb 2021 → recovery
Nov 2024, 982 days) make clear this is **not** a market-neutral strategy —
it's a long-only, beta-heavy momentum/trend tilt that outperformed on a raw
return basis but carries meaningfully more risk than the benchmark. Round-trip
analysis: 1,597 trades, 61% win rate.

### Alphalens factor diagnostics (on the XGBoost `factor_signal`)

| Horizon | IC mean | Risk-adj. IC | Top-quantile mean return | Bottom-quantile mean return |
|---|---|---|---|---|
| 5D | 0.043 | 0.261 | 18.8 bps | −10.4 bps |
| 10D | 0.043 | 0.260 | 19.0 bps | −10.5 bps |
| 21D | 0.042 | 0.256 | 19.2 bps | −10.3 bps |
| 63D | 0.047 | 0.290 | 13.6 bps | −6.1 bps |

t-stat(IC) at the 63-day horizon is 18.45 (p≈0) — the ranking signal is
statistically real. The caveat is **turnover**: mean turnover in the
extreme quantiles runs from ~18% (5D) up to ~93% (63D), which is expensive
to trade and is a large part of why the realized Sharpe (0.83) is much
lower than the raw IC would suggest in isolation.

## Honest limitations

- **BH-FDR wiped out every "significant" feature.** Naive significance
  testing at p<0.05 flagged 4 features; none survived multiple-testing
  correction. The 8 features that made the final cut got there via the IC
  threshold + clustering path, not FDR — worth stating plainly rather than
  implying the selection is FDR-validated.
- **3 of 8 selected features are unstable** under block-bootstrap resampling
  (see table above) and should not be trusted at face value without further
  investigation.
- **The two ML importance methods disagree completely** (Spearman = 0.000),
  so "consensus top features" language should be read as "same features,
  no agreement on order," not as independent confirmation.
- **High turnover** in the ranking signal (up to 93% at the 63-day horizon)
  means the backtest's realized performance already reflects real trading
  costs via Zipline's execution model, but a live implementation would need
  its own slippage/capacity analysis before sizing this beyond the backtest's
  universe.
- **Universe size dips to 1 name** in the earliest rebalance dates — an
  artifact of the point-in-time history requirement rather than a real
  market condition, and should be excluded or flagged before using early
  history for anything beyond illustration.

## How to run

This pipeline depends on internal course/lab infrastructure
(`qsconnect`, `qsresearch`, `qsbacktest`) for data access, indicator
computation, and the Zipline-based backtest harness — those packages are
licensed lab material and are not included or redistributed here. What's
original and reproducible is the **selection methodology**: the IC/
correlation/clustering/BH-FDR/bootstrap/ML-importance pipeline in
`src/feature_selection.py` operates on a plain pandas/polars DataFrame with
`symbol`, `date`, `close`, `high`, `low`, `volume` columns and no
proprietary dependency once your own OHLCV + indicator columns are supplied.

```bash
git clone https://github.com/[YOUR-USERNAME]/dynamic-universe-factor-selection.git
cd dynamic-universe-factor-selection
pip install -r requirements.txt

# Supply your own OHLCV panel with symbol/date/open/high/low/close/volume,
# and your own indicator columns (or substitute a public library like
# ta-lib / pandas-ta for the technical indicators used here).
python src/build_universe.py        # point-in-time eligibility + reduction
python src/feature_selection.py     # IC -> correlation -> clustering -> BH-FDR -> bootstrap -> ML importance
python src/generate_report.py       # selection summary + charts in /results
```

## What I'd do next

- Drop `atr_extension_200`, `close_roc_0_5`, and `close_roc_0_21` from the
  "confirmed" set given their bootstrap sign instability, and re-run the
  backtest on the 5 stable survivors alone to see how much of the Sharpe
  ratio they were actually contributing.
- Investigate *why* MDI and permutation importance disagree so completely —
  likely candidates are feature collinearity interacting with tree splits,
  or the small final feature count making rank orderings noisy.
- Add an explicit transaction-cost/capacity model on top of Zipline's
  default execution assumptions, given the turnover profile at longer
  horizons.
- Test industry/sector neutrality on the ranking signal — none of the 8
  features are sector-aware, so part of the realized 1.28 beta may be a
  disguised sector or size tilt rather than genuine stock selection.
- Re-run the eligibility filter with a hard floor on universe size per date
  to eliminate the early min-of-1 artifact.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── build_universe.py       # point-in-time eligibility + symbol reduction
│   ├── feature_selection.py    # IC / correlation / clustering / BH-FDR / bootstrap / ML importance
│   └── generate_report.py      # selection summary + charts
├── results/
│   ├── ic_ranking.png
│   ├── correlation_clustered.png
│   ├── stability_selection.png
│   └── selection_report.txt
└── notebooks/
    └── exploration.ipynb
```

## Notes on originality

The data ingestion and backtest execution layer (`qsconnect`/`qsresearch`/
`qsbacktest`) is licensed lab infrastructure and is not published here. The
feature-selection methodology — the ordering of the funnel, the Newey-West
IC screen, the correlation/clustering deduplication, the BH-FDR check, the
block-bootstrap stability selection, and the ML importance cross-check — is
original analysis logic applying standard published techniques (Newey &
West 1987 HAC estimation; Benjamini & Hochberg 1995 FDR control) to an
original universe and feature set.
