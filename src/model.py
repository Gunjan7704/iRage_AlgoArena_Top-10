"""
Model training and ensemble pipeline for iRage AlgoArena 2026.

This module implements:
1. LightGBM parameter configurations (high vs low regularization)
2. Single-fold training with early stopping
3. GroupKFold cross-validation respecting temporal structure
4. Weighted ensemble of conservative and aggressive models
"""

from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_lgbm_params(regularization: str = "high") -> Dict:
    """Return LightGBM hyperparameters for the specified regularization level.

    Two configurations are used to produce decorrelated predictions:
    - 'high' (Model A): Strong L1/L2 penalties (α=5, λ=5) → smoother,
      more robust predictions with lower variance. Acts as the stable
      backbone of the ensemble.
    - 'low' (Model B): Weak L1/L2 penalties (α=0.5, λ=0.5) → more
      expressive fit that captures finer-grained patterns, at the cost
      of higher variance.

    All other hyperparameters are shared to isolate regularization as
    the only source of model diversity.

    Args:
        regularization: Either 'high' (conservative) or 'low' (aggressive).

    Returns:
        Dictionary of LightGBM parameters.

    Raises:
        ValueError: If regularization is not 'high' or 'low'.
    """
    if regularization not in ("high", "low"):
        raise ValueError(f"regularization must be 'high' or 'low', got '{regularization}'")

    reg_values = {"high": (5.0, 5.0), "low": (0.5, 0.5)}
    alpha, lam = reg_values[regularization]

    return {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 128,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": alpha,
        "reg_lambda": lam,
        "random_state": 42,
        "verbose": -1,
    }


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM model on one CV fold with early stopping.

    Uses L2 (MSE) as the evaluation metric and stops training if the
    validation score does not improve for 200 consecutive rounds.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        X_val: Validation feature matrix.
        y_val: Validation target array.
        params: LightGBM hyperparameter dictionary (from get_lgbm_params).

    Returns:
        Trained LGBMRegressor model.
    """
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=500),
        ],
    )

    return model


def run_cv(
    train: pd.DataFrame,
    features: List[str],
    target: str,
    group_col: str,
    params: Dict,
    test: Optional[pd.DataFrame] = None,
    n_splits: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run GroupKFold cross-validation and return OOF and test predictions.

    GroupKFold splits ensure that all rows sharing the same CV_GROUP
    appear entirely in either the training or validation set. This
    prevents temporal leakage — a critical requirement for financial
    data where consecutive time periods are autocorrelated.

    Predictions are averaged across all folds for the test set.

    Args:
        train: Training dataframe with features, target, and group column.
        features: List of feature column names to use.
        target: Name of the target column.
        group_col: Name of the group column for GroupKFold.
        params: LightGBM hyperparameter dictionary.
        test: Optional test dataframe. If provided, test predictions
              are generated and averaged across folds.
        n_splits: Number of CV folds.

    Returns:
        Tuple of (oof_predictions, test_predictions).
        test_predictions is None if no test dataframe is provided.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = train[group_col].values

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test)) if test is not None else None

    X_all = train[features].values
    y_all = train[target].values
    X_test = test[features].values if test is not None else None

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"  Train size: {len(train_idx):,}  |  Val size: {len(val_idx):,}")
        print(f"{'='*60}")

        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        model = train_fold(X_tr, y_tr, X_val, y_val, params)

        # Out-of-fold predictions
        oof_preds[val_idx] = model.predict(X_val)

        # Accumulate test predictions (will average later)
        if X_test is not None:
            test_preds += model.predict(X_test)

    # Average test predictions across folds
    if test_preds is not None:
        test_preds /= n_splits

    # Report overall OOF score
    oof_mse = np.mean((oof_preds - y_all) ** 2)
    print(f"\nOverall OOF MSE: {oof_mse:.6f}")

    return oof_preds, test_preds


def ensemble(
    pred_A: np.ndarray,
    pred_B: np.ndarray,
    weight_A: float = 0.72,
) -> np.ndarray:
    """Blend predictions from two models using a weighted average.

    The default 72:28 weighting favours the conservative (high-reg)
    model. In financial prediction, stability matters more than
    sharpness — overfit predictions don't just lose accuracy, they
    produce catastrophic outliers. The 28% allocation to the
    aggressive model injects additional signal without letting
    the higher-variance model dominate.

    Args:
        pred_A: Predictions from Model A (conservative, high regularization).
        pred_B: Predictions from Model B (aggressive, low regularization).
        weight_A: Weight for Model A. Model B receives (1 - weight_A).

    Returns:
        Blended prediction array.
    """
    weight_B = 1.0 - weight_A
    blended = weight_A * pred_A + weight_B * pred_B
    print(f"Ensemble: {weight_A:.0%} Model A + {weight_B:.0%} Model B")
    return blended
