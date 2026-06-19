# Feature Engineering Summary

## Feature Categories

| Category | Count | Description |
|---|---|---|
| **Raw (selected)** | 120 | Top features by \|Pearson correlation\| with TARGET |
| **Momentum** | varies* | `LagT1 − LagT2` for each signal with multiple lags |
| **Acceleration** | varies* | `LagT1 − 2·LagT2 + LagT3` for signals with 3+ lags |
| **Intra-signal interactions** | 190 | Pairwise products of top 20 lag features (C(20,2)) |
| **Cross-signal interactions** | 50 | S01×S02 (25) + S01×S03 (25), top 5 features per group |

*\*Momentum and acceleration counts depend on how many selected features have matching LagT2/LagT3 columns in the dataset.*

---

## Why Each Category Was Included

### Raw Features (120)

The raw feature space contains hundreds of signal columns, most of which are noise. Computing `|corr(feature, TARGET)|` and retaining only the top 120 eliminates uninformative features early. This prevents tree-based models from wasting splits on noise and reduces computation for all downstream feature engineering steps.

### Momentum Features

`momentum = LagT1 - LagT2`

The first discrete difference captures the **rate of change** of a signal between the two most recent observations. In financial markets, the *direction* and *speed* of signal movement is often more predictive than the signal's absolute level. A signal at value 5 that was at 3 yesterday carries different information than one at 5 that was at 7 yesterday.

### Acceleration Features

`acceleration = LagT1 - 2·LagT2 + LagT3`

The second discrete difference captures the **change in momentum** — whether a signal's trend is strengthening, stable, or fading. This is the discrete analogue of the second derivative. A decelerating uptrend (positive momentum, negative acceleration) suggests a regime change may be approaching. This mirrors how traders and quantitative strategies interpret price dynamics.

### Intra-Signal Interaction Features (190)

Pairwise products of the top 20 lag features. Tree models find axis-aligned decision boundaries, but real predictive signal in financial data often lies along diagonals in feature space. Products allow the model to capture **multiplicative conditional relationships** — e.g., a feature that is only informative when another feature exceeds a threshold. With 20 features, C(20,2) = 190 interactions remain computationally tractable.

### Cross-Signal Interaction Features (50)

Products between the strongest features of different signal families (S01×S02 and S01×S03). Different signal families likely encode different aspects of market microstructure — order flow imbalance, volatility, bid-ask dynamics, etc. Cross-products let the model detect **joint regimes** where combinations of signals are predictive in ways that isolated signals are not. Limiting to 5 features per group (5×5 = 25 per pair, 50 total) keeps dimensionality controlled.

---

## Results

![Leaderboard](leaderboard.jpeg)
