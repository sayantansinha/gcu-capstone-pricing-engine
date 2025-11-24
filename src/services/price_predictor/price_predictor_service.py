from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import joblib
import numpy as np
import pandas as pd

from src.utils.data_io_utils import save_processed
from src.utils.log_utils import get_logger
from src.utils.model_io_utils import (
    load_model_registry,
    _get_model_store,
    _LocalModelStore,
    _S3ModelStore,
)

LOGGER = get_logger("predict_price_service")


# ---------------------------------------------------------------------
# Registry model option (for dropdown)
# ---------------------------------------------------------------------
@dataclass
class RegistryModelOption:
    run_id: str
    model_name: str
    status: str
    created_at: datetime
    metrics: Dict[str, Any]

    @property
    def label(self) -> str:
        rmse = None
        metrics = self.metrics or {}
        for k in ("rmse", "RMSE", "val_rmse"):
            if k in metrics:
                rmse = metrics[k]
                break

        status_part = self.status or "unknown"
        if rmse is not None:
            return f"{self.model_name} [{status_part}, RMSE={rmse:,.1f}]"
        return f"{self.model_name} [{status_part}]"


# ---------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------
def get_registry_model_options(
        status_filter: Optional[List[str]] = None,
) -> List[RegistryModelOption]:
    """
    Load global model registry → dropdown options.
    Sorted by created_at DESC so index 0 is latest.
    """
    raw_entries = load_model_registry()  # list[dict]
    if not raw_entries:
        LOGGER.info("Model registry empty; no models available for prediction.")
        return []

    options: List[RegistryModelOption] = []
    for e in raw_entries:
        status = e.get("status", "unknown")
        if status_filter and status not in status_filter:
            continue

        created_raw = e.get("created_at") or e.get("created") or ""
        try:
            created_at = datetime.fromisoformat(created_raw.replace("Z", ""))
        except Exception:
            created_at = datetime.min

        opt = RegistryModelOption(
            run_id=e.get("run_id", ""),
            model_name=e.get("model_name", ""),
            status=status,
            created_at=created_at,
            metrics=e.get("metrics", {}) or {},
        )
        if opt.run_id and opt.model_name:
            options.append(opt)

    options.sort(key=lambda o: o.created_at, reverse=True)
    return options


# ---------------------------------------------------------------------
# Model loading (from MODELS_DIR or MODELS_BUCKET)
# ---------------------------------------------------------------------
def load_stacked_model(option: RegistryModelOption):
    """
    Load the stacked/ensemble model artifact associated with a registry entry.

    Assumes model was saved as `<model_name>.joblib` under that run’s model store.
    """
    store = _get_model_store(option.run_id)
    filename = f"{option.model_name}.joblib"

    # LOCAL backend
    if isinstance(store, _LocalModelStore):
        root = getattr(store, "root", None)
        if root is None:
            raise RuntimeError("LocalModelStore has no 'root' attribute; cannot locate model artifact.")

        path = Path(root) / filename
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        LOGGER.info("Loading stacked model (LOCAL) from %s", path)
        return joblib.load(path)

    # S3 backend
    if isinstance(store, _S3ModelStore):
        bucket = getattr(store, "bucket", None)
        prefix = getattr(store, "prefix", "")
        if not bucket:
            raise RuntimeError("S3ModelStore has no bucket configured; cannot load model artifact.")

        key = f"{prefix}{filename}"
        uri = f"s3://{bucket}/{key}"
        LOGGER.info("Loading stacked model (S3) from %s", uri)

        with fsspec.open(uri, "rb") as f:
            return joblib.load(f)

    raise RuntimeError(f"Unsupported model store type for run_id={option.run_id}: {type(store)}")


# ---------------------------------------------------------------------
# File upload handling
# ---------------------------------------------------------------------
def save_uploaded_assets_df(df: pd.DataFrame, original_name: str) -> str:
    """
    Save uploaded prediction dataset into a separate logical directory
    (NOT the pipeline raw/processed/source run dirs).

    Uses processed backend with base_dir='predict_inputs'.
    """
    base_name = Path(original_name).stem or "predict_assets"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{timestamp}_{base_name}"

    # LOCAL: <PROCESSED_DIR>/predict_inputs/<timestamp>_<base>.parquet
    # S3:    s3://<PROCESSED_BUCKET>/predict_inputs/<timestamp>_<base>.parquet
    saved_path = save_processed(df, base_dir="predict_inputs", name=name)
    LOGGER.info("Saved uploaded prediction input → %s", saved_path)
    return saved_path


# ---------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------
def _get_feature_names_for_model(model: Any) -> list[str]:
    """
    Infer training feature names for a model.

    - If the model itself exposes feature_names_in_, use that.
    - If it's a StackedEnsembleRegressor, look inside its base_models.
      Those are the estimators trained on the original feature matrix.
    """
    # 1. Plain sklearn estimator
    feat_names = getattr(model, "feature_names_in_", None)
    if feat_names is not None:
        # sklearn exposes this as a numpy array
        return list(feat_names)

    # 2. Our stacked ensemble wrapper
    base_models = getattr(model, "base_models", None)
    if isinstance(base_models, dict) and base_models:
        for est in base_models.values():
            # Some code paths might store a list of models per name (e.g. OOF),
            # so pick the first element if it's a list.
            if isinstance(est, (list, tuple)):
                if est and hasattr(est[0], "feature_names_in_"):
                    return list(est[0].feature_names_in_)
            else:
                if hasattr(est, "feature_names_in_"):
                    return list(est.feature_names_in_)

    raise ValueError(
        "Could not infer feature_names_in_ from the stacked model or its base "
        "estimators. Manual single-asset prediction cannot construct the feature row."
    )


def build_manual_feature_row(model: Any, asset_features: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from raw manual inputs, matching
    the model's training feature names (model.feature_names_in_).

    Expected raw keys in asset_features:
      - territory: e.g. "USA"
      - media: e.g. "SVOD"
      - platform: e.g. "NETFLIX"
      - exclusivity: e.g. "EXCLUSIVE"  (currently ignored unless you added exclusivity_* features)
      - release_year: int
      - imdb_votes: int
      - units: numeric (optional; defaults to 1 if used by model)
    """
    feature_names = _get_feature_names_for_model(model)
    cols = list(feature_names)
    X = pd.DataFrame(columns=cols, index=[0], dtype=float)
    X.loc[0] = 0.0  # default all to 0

    # numeric: release_year
    if "release_year" in cols and "release_year" in asset_features:
        X.at[0, "release_year"] = float(asset_features["release_year"])

    # numeric: units (if used in model)
    if "units" in cols:
        X.at[0, "units"] = float(asset_features.get("units", 1.0))

    # numeric: log1p_numVotes from imdb_votes / numVotes
    if "log1p_numVotes" in cols:
        votes = asset_features.get("imdb_votes") or asset_features.get("numVotes") or 0
        X.at[0, "log1p_numVotes"] = float(np.log1p(votes))

    # one-hot: territory_<CODE>
    territory_code = asset_features.get("territory")
    if territory_code is not None:
        col = f"territory_{territory_code}"
        if col in cols:
            X.at[0, col] = 1.0

    # one-hot: media_<CODE>
    media_code = asset_features.get("media")
    if media_code is not None:
        col = f"media_{media_code}"
        if col in cols:
            X.at[0, col] = 1.0

    # one-hot: platform_<CODE>
    platform_code = asset_features.get("platform")
    if platform_code is not None:
        col = f"platform_{platform_code}"
        if col in cols:
            X.at[0, col] = 1.0

    return X


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------
def predict_for_dataframe(
        model,
        df_features: pd.DataFrame,
        price_column_name: str = "predicted_price",
) -> pd.DataFrame:
    """
    Run predictions for a feature DataFrame using the loaded model.

    Assumes df_features already matches the training feature schema.
    """
    if df_features.empty:
        raise ValueError("Input DataFrame for prediction is empty.")

    preds = model.predict(df_features)
    out = df_features.copy()
    out[price_column_name] = preds
    return out


def predict_for_single_asset(
        model,
        asset_features: Dict[str, Any],
        price_column_name: str = "predicted_price",
) -> Tuple[float, pd.DataFrame]:
    """
    Run a prediction for a single asset from raw manual inputs:
    builds the engineered feature row, then calls model.predict().
    """
    X = build_manual_feature_row(model, asset_features)
    preds = model.predict(X)
    out_df = X.copy()
    out_df[price_column_name] = preds
    price = float(preds[0])
    return price, out_df


# ---------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------
def compute_confidence_interval_and_score(
        price: float,
        metrics: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Derive a simple confidence interval and confidence score from model metrics.

    Uses RMSE (or RMSE-like) metric as a proxy for error spread:
      - CI ≈ price ± 1.96 * rmse
      - confidence score ≈ max(0, 1 - rmse / |price|) in [0, 100]

    Returns:
        (ci_low, ci_high, confidence_percent)
    """
    if metrics is None:
        return None, None, None

    rmse = None
    for key in ("rmse", "RMSE", "val_rmse"):
        if key in metrics:
            try:
                rmse = float(metrics[key])
                break
            except (TypeError, ValueError):
                continue

    if rmse is None or rmse <= 0:
        return None, None, None

    ci_low = price - 1.96 * rmse
    ci_high = price + 1.96 * rmse

    if price == 0:
        confidence = None
    else:
        confidence = max(0.0, min(0.99, 1.0 - rmse / abs(price))) * 100.0

    return ci_low, ci_high, confidence
