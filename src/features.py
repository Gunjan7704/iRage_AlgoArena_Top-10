"""
Feature engineering pipeline for iRage AlgoArena 2026.

This module implements all feature engineering steps:
1. Correlation-based feature selection
2. Momentum features (first discrete difference)
3. Acceleration features (second discrete difference)
4. Intra-signal pairwise interaction features
5. Cross-signal (S01×S02, S01×S03) interaction features
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def select_top_features(
    train: pd.DataFrame,
    target_col: str = "TARGET",
    n: int = 120,
) -> List[str]:
    """Select the top-n features by absolute Pearson correlation with target.

    Univariate correlation filtering removes noise features before
    downstream engineering. In financial data, most raw signals are
    uninformative — retaining only high-correlation features keeps the
    signal-to-noise ratio high and prevents tree-based models from
    wasting splits on irrelevant columns.

    Args:
        train: Training dataframe containing signal features and target.
        target_col: Name of the target column.
        n: Number of top features to retain.

    Returns:
        List of feature names sorted by descending |correlation| with target.
    """
    # Exclude non-feature columns
    exclude_cols = {"ID", target_col, "CV_GROUP"}
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    # Compute absolute Pearson correlation with target
    correlations = train[feature_cols].corrwith(train[target_col]).abs()
    correlations = correlations.sort_values(ascending=False)

    top_features = correlations.head(n).index.tolist()
    print(f"Selected top {len(top_features)} features by |correlation| with {target_col}")
    return top_features


def build_momentum_features(
    df: pd.DataFrame,
    top_features: List[str],
) -> Dict[str, np.ndarray]:
    """Build momentum (first difference) features from lag structure.

    For every LagT1 feature in the selected set, if LagT2 and LagT3
    also exist in the full dataset:
        momentum = LagT1 - LagT2  (rate of change)
        acceleration = LagT1 - 2*LagT2 + LagT3  (change in rate of change)

    Financial intuition: raw lag values tell you WHERE a signal is;
    differences tell you WHERE IT'S GOING. Momentum captures trend
    direction, acceleration captures whether the trend is strengthening
    or fading — analogous to how traders read price action.

    Args:
        df: Dataframe containing the raw features.
        top_features: List of selected feature names.

    Returns:
        Dictionary mapping new feature names to their computed arrays.
    """
    new_features: Dict[str, np.ndarray] = {}

    # Identify LagT1 features among the top selected features
    lag1_features = [f for f in top_features if "_LagT1" in f]

    for feat in lag1_features:
        base = feat.replace("_LagT1", "")
        lag2_name = f"{base}_LagT2"
        lag3_name = f"{base}_LagT3"

        # Only build if the required lags exist in the dataframe
        if lag2_name in df.columns:
            # First discrete difference: rate of change
            momentum_name = f"momentum_{base}"
            new_features[momentum_name] = df[feat].values - df[lag2_name].values

        if lag2_name in df.columns and lag3_name in df.columns:
            # Second discrete difference: change in momentum
            acceleration_name = f"accel_{base}"
            new_features[acceleration_name] = (
                df[feat].values - 2 * df[lag2_name].values + df[lag3_name].values
            )

    print(f"Built {len(new_features)} momentum/acceleration features")
    return new_features


def build_intra_interactions(
    df: pd.DataFrame,
    top_features: List[str],
    n: int = 20,
) -> Dict[str, np.ndarray]:
    """Build pairwise product features within the top-n lag features.

    Takes the first n features from the sorted top_features list and
    computes all C(n, 2) pairwise products. For n=20, this yields 190
    interaction features.

    Financial intuition: tree models find axis-aligned splits, but
    predictive signal boundaries in financial data often lie along
    diagonals in feature space. Pairwise products capture multiplicative
    relationships — e.g., a momentum signal that is only informative
    when a volatility signal is elevated.

    Args:
        df: Dataframe containing the raw features.
        top_features: List of selected feature names (sorted by importance).
        n: Number of top features to use for interactions.

    Returns:
        Dictionary mapping interaction feature names to their computed arrays.
    """
    interaction_features: Dict[str, np.ndarray] = {}
    top_n = top_features[:n]

    for i in range(len(top_n)):
        for j in range(i + 1, len(top_n)):
            f1, f2 = top_n[i], top_n[j]
            name = f"intra_{f1}_x_{f2}"
            interaction_features[name] = df[f1].values * df[f2].values

    print(f"Built {len(interaction_features)} intra-signal interaction features")
    return interaction_features


def build_cross_interactions(
    df: pd.DataFrame,
    top_features: List[str],
    n_per_group: int = 5,
) -> Dict[str, np.ndarray]:
    """Build cross-signal interaction features between S01, S02, and S03.

    Takes the top n_per_group features from each signal family (S01, S02, S03)
    among the selected features and computes pairwise cross-products:
        - S01 × S02 = n_per_group² features
        - S01 × S03 = n_per_group² features

    Financial intuition: different signal families likely capture different
    market microstructure effects (order flow, volatility, spread dynamics).
    Cross-signal products allow the model to detect regimes where combinations
    of signals are jointly predictive — something isolated signals cannot express.

    Args:
        df: Dataframe containing the raw features.
        top_features: List of selected feature names.
        n_per_group: Number of top features to select from each signal group.

    Returns:
        Dictionary mapping cross-interaction feature names to their computed arrays.
    """
    cross_features: Dict[str, np.ndarray] = {}

    # Extract top features per signal family, preserving the correlation-sorted order
    s01_feats = [f for f in top_features if f.startswith("S01_")][:n_per_group]
    s02_feats = [f for f in top_features if f.startswith("S02_")][:n_per_group]
    s03_feats = [f for f in top_features if f.startswith("S03_")][:n_per_group]

    # S01 × S02 cross interactions
    for f1 in s01_feats:
        for f2 in s02_feats:
            name = f"cross12_{f1}_x_{f2}"
            cross_features[name] = df[f1].values * df[f2].values

    # S01 × S03 cross interactions
    for f1 in s01_feats:
        for f2 in s03_feats:
            name = f"cross13_{f1}_x_{f2}"
            cross_features[name] = df[f1].values * df[f2].values

    print(f"Built {len(cross_features)} cross-signal interaction features")
    return cross_features


def engineer_all_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Run the full feature engineering pipeline on train and test sets.

    Applies momentum, acceleration, intra-signal interactions, and
    cross-signal interactions to both datasets consistently.

    Args:
        train: Training dataframe.
        test: Test dataframe.
        top_features: Pre-selected top features (from select_top_features).

    Returns:
        Tuple of (engineered_train, engineered_test, feature_list) where
        feature_list contains all feature column names to use for modelling.
    """
    train_eng = train.copy()
    test_eng = test.copy()

    feature_list = list(top_features)  # Start with selected raw features

    # --- Momentum & Acceleration ---
    for label, df in [("train", train_eng), ("test", test_eng)]:
        momentum_feats = build_momentum_features(df, top_features)
        for feat_name, values in momentum_feats.items():
            df[feat_name] = values
            if label == "train" and feat_name not in feature_list:
                feature_list.append(feat_name)

    # --- Intra-signal Interactions ---
    for label, df in [("train", train_eng), ("test", test_eng)]:
        intra_feats = build_intra_interactions(df, top_features, n=20)
        for feat_name, values in intra_feats.items():
            df[feat_name] = values
            if label == "train" and feat_name not in feature_list:
                feature_list.append(feat_name)

    # --- Cross-signal Interactions ---
    for label, df in [("train", train_eng), ("test", test_eng)]:
        cross_feats = build_cross_interactions(df, top_features, n_per_group=5)
        for feat_name, values in cross_feats.items():
            df[feat_name] = values
            if label == "train" and feat_name not in feature_list:
                feature_list.append(feat_name)

    print(f"\nTotal engineered features: {len(feature_list)}")
    return train_eng, test_eng, feature_list
