from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from statsmodels.stats.diagnostic import het_breuschpagan
from xgboost import XGBRegressor

from src.model.stacked_ensembler import StackedEnsembleRegressor

# ---------------------------------------------------------------------
# Allowed models for creating the stacked ensembler
# ---------------------------------------------------------------------
AVAILABLE_MODELS: Dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
    "MLP": MLPRegressor,
}


# ---------------------------------------------------------------------
# Meta-Stacked Ensembler
# ---------------------------------------------------------------------
def build_stacked_ensemble(
        base_out: Dict[str, Any],
        meta_cls=Ridge,
        meta_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train a stacked ensemble on the validation predictions of the base models.
    """
    meta_params = meta_params or {}

    yv = base_out["y_valid"]
    base_models = base_out["models"]  # dict: model_name -> fitted estimator

    # Keep a stable order of base models
    names = list(base_out["valid_preds"].keys())
    M_valid = np.column_stack([base_out["valid_preds"][n] for n in names])

    # Train meta-model on validation preds
    meta_model = meta_cls(**meta_params)
    meta_model.fit(M_valid, yv)

    # Validation predictions of stacked ensemble
    stacked_valid_pred = meta_model.predict(M_valid)

    # Wrap base + meta in a single predictor object
    ensemble_model = StackedEnsembleRegressor(
        base_models=base_models,
        meta_model=meta_model,
        base_model_order=names,
    )

    return {
        "kind": "stacked",
        "pred": stacked_valid_pred,
        "metrics": _metrics(yv, stacked_valid_pred),
        "base_model_names": names,
        "meta_model": meta_model,
        "model": ensemble_model,  # <-- this is what you'll persist for inference
    }


# ---------------------------------------------------------------------
# Other Ensembles (kept for comparison)
# ---------------------------------------------------------------------
def combine_average(base_out: Dict[str, Any]) -> Dict[str, Any]:
    yv = base_out["y_valid"]
    M = np.column_stack(list(base_out["valid_preds"].values()))
    pred = M.mean(axis=1)
    return {"kind": "average", "pred": pred, "metrics": _metrics(yv, pred)}


def combine_weighted_inverse_rmse(base_out: Dict[str, Any]) -> Dict[str, Any]:
    yv = base_out["y_valid"]
    metrics = {r["model"]: r for r in base_out["per_model_metrics"]}
    names = list(base_out["valid_preds"].keys())
    M = np.column_stack([base_out["valid_preds"][n] for n in names])
    rmses = np.array([metrics[n]["RMSE"] for n in names])
    inv = 1.0 / (rmses + 1e-8)
    w = inv / inv.sum()
    pred = (M * w).sum(axis=1)
    return {
        "kind": "weighted_inverse_rmse",
        "pred": pred,
        "weights": {n: float(w[i]) for i, n in enumerate(names)},
        "metrics": _metrics(yv, pred)
    }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def _split(df: pd.DataFrame, target: str, test_size: float = 0.2, rs: int = 42):
    X = df.drop(columns=[target])
    y = df[target].values
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=rs)
    return Xtr, Xva, ytr, yva, X.columns.tolist()


def _build_model(name: str, params: Dict[str, Any]):
    cls = AVAILABLE_MODELS.get(name)
    if cls is None:
        raise ValueError(f"Model {name} is not supported.")
    if cls is XGBRegressor and XGBRegressor is None:
        raise ImportError("XGBoost not installed.")
    if cls is LGBMRegressor and LGBMRegressor is None:
        raise ImportError("LightGBM not installed.")
    return cls(**(params or {}))


def _breusch_pagan_for_linear(X_va: pd.DataFrame, y_va: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    try:
        residuals = y_va - y_pred
        X_sm = sm.add_constant(X_va, has_constant="add")
        bp = het_breuschpagan(residuals, X_sm)
        return {"LM": float(bp[0]), "LM_pvalue": float(bp[1]), "F": float(bp[2]), "F_pvalue": float(bp[3])}
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Training base models (evaluate individually)
# ---------------------------------------------------------------------
def train_base_models(
        df: pd.DataFrame,
        target: str,
        model_names: List[str],
        params_map: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Train and evaluate multiple models on a consistent train/validation split.

    Returns:
        {
            "features": [str],              # feature names
            "X_valid": pd.DataFrame,        # NEW: validation feature matrix (for explainability)
            "y_valid": np.ndarray,          # validation target
            "valid_preds": {name: np.ndarray},
            "per_model_metrics": [ {...}, ... ],
            "models": {name: fitted_model},
        }
    """
    Xtr, Xva, ytr, yva, feat_names = _split(df, target)
    params_map = params_map or {}
    per_model_metrics: List[Dict[str, Any]] = []
    valid_preds: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Any] = {}

    # Keep validation features as a DataFrame so downstream code (explainability, BP test, etc.)
    # has easy access to column names.
    Xva_df = pd.DataFrame(Xva, columns=feat_names)

    for name in model_names:
        mdl = _build_model(name, params_map.get(name, {}))
        mdl.fit(Xtr, ytr)
        pr = mdl.predict(Xva)
        valid_preds[name] = pr
        fitted[name] = mdl

        row = {"model": name, **_metrics(yva, pr)}
        # Compute BP for linear-family models
        if name in ("LinearRegression", "Ridge", "Lasso"):
            row["bp"] = _breusch_pagan_for_linear(Xva_df, yva, pr)
        per_model_metrics.append(row)

    return {
        "features": feat_names,
        "X_valid": Xva_df,  # NEW: used by explainability + reports
        "y_valid": yva,
        "valid_preds": valid_preds,
        "per_model_metrics": per_model_metrics,
        "models": fitted,
    }


# ---------------------------------------------------------------------
# Ensembles (computed automatically; no UI selection)
# ---------------------------------------------------------------------
def combine_average(base_out: Dict[str, Any]) -> Dict[str, Any]:
    yv = base_out["y_valid"]
    M = np.column_stack(list(base_out["valid_preds"].values()))
    pred = M.mean(axis=1)
    return {"kind": "average", "pred": pred, "metrics": _metrics(yv, pred)}


def combine_weighted_inverse_rmse(base_out: Dict[str, Any]) -> Dict[str, Any]:
    yv = base_out["y_valid"]
    metrics = {r["model"]: r for r in base_out["per_model_metrics"]}
    names = list(base_out["valid_preds"].keys())
    M = np.column_stack([base_out["valid_preds"][n] for n in names])
    rmses = np.array([metrics[n]["RMSE"] for n in names])
    inv = 1.0 / (rmses + 1e-8)
    w = inv / inv.sum()
    pred = (M * w).sum(axis=1)
    return {
        "kind": "weighted_inverse_rmse",
        "pred": pred,
        "weights": {n: float(w[i]) for i, n in enumerate(names)},
        "metrics": _metrics(yv, pred)
    }
