import io
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from src.config.env_loader import SETTINGS
from src.utils.data_io_utils import _ensure_buckets, is_s3
from src.utils.log_utils import get_logger
from src.utils.s3_utils import (
    write_bucket_object,
    formulate_s3_uri,
    list_bucket_objects,
    load_bucket_object,
)

LOGGER = get_logger("model_io_utils")


# -------------------------------------------------------------------------
# Low-level S3 JSON helper (still useful for registry + artifacts)
# -------------------------------------------------------------------------
def _write_json_to_bucket(bucket: str, key_prefix: str, filename: str, obj: Dict[str, Any]) -> None:
    """
    Small helper to write a JSON object to S3 under the given prefix/filename.
    """
    write_bucket_object(
        bucket,
        f"{key_prefix}{filename}",
        json.dumps(obj, indent=2).encode("utf-8"),
        content_type="application/json",
    )


# -------------------------------------------------------------------------
# ModelStore abstraction: hides LOCAL vs S3 differences
# -------------------------------------------------------------------------
class _BaseModelStore:
    """
    Simple abstraction for model artifact storage.
    Concrete implementations: _LocalModelStore and _S3ModelStore.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id.strip("/")

    # --- Saving ---
    def save_csv(self, filename: str, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def save_json(self, filename: str, obj: Dict[str, Any]) -> None:
        raise NotImplementedError

    def save_joblib(self, filename: str, obj: Any) -> None:
        raise NotImplementedError

    # --- Loading ---
    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def load_json(self, filename: str) -> Optional[dict]:
        raise NotImplementedError

    # --- Misc ---
    def run_exists(self) -> bool:
        raise NotImplementedError

    def uri(self) -> str:
        raise NotImplementedError


class _LocalModelStore(_BaseModelStore):
    """
    Local filesystem-backed model storage.
    """

    def __init__(self, run_id: str):
        super().__init__(run_id)
        models_root = Path(getattr(SETTINGS, "MODELS_DIR", "models"))
        self.root = models_root / self.run_id

    def _ensure_dir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def save_csv(self, filename: str, df: pd.DataFrame) -> None:
        self._ensure_dir()
        df.to_csv(self.root / filename, index=False)

    def save_json(self, filename: str, obj: Dict[str, Any]) -> None:
        self._ensure_dir()
        with open(self.root / filename, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def save_joblib(self, filename: str, obj: Any) -> None:
        self._ensure_dir()
        joblib.dump(obj, self.root / filename)

    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        path = self.root / filename
        if not path.exists():
            LOGGER.info(f"load_model_csv(LOCAL) missing: {path}")
            return None
        return pd.read_csv(path)

    def load_json(self, filename: str) -> Optional[dict]:
        path = self.root / filename
        if not path.exists():
            LOGGER.info(f"load_model_json(LOCAL) missing: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_exists(self) -> bool:
        exists = self.root.exists() and any(self.root.iterdir())
        LOGGER.info(f"model_run_exists(LOCAL) run_id={self.run_id} → {exists}")
        return exists

    def uri(self) -> str:
        return str(self.root)


class _S3ModelStore(_BaseModelStore):
    """
    S3-backed model storage.
    """

    def __init__(self, run_id: str):
        super().__init__(run_id)
        bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
        if not bucket:
            raise ValueError("MODELS_BUCKET not configured for S3 backend")
        self.bucket = bucket
        self.prefix = f"{self.run_id}/"

    def _ensure_bucket(self) -> None:
        _ensure_buckets()

    def save_csv(self, filename: str, df: pd.DataFrame) -> None:
        self._ensure_bucket()
        write_bucket_object(
            self.bucket,
            f"{self.prefix}{filename}",
            df.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
        )

    def save_json(self, filename: str, obj: Dict[str, Any]) -> None:
        self._ensure_bucket()
        _write_json_to_bucket(self.bucket, self.prefix, filename, obj)

    def save_joblib(self, filename: str, obj: Any) -> None:
        self._ensure_bucket()
        buf = BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        write_bucket_object(
            self.bucket,
            f"{self.prefix}{filename}",
            buf.read(),
            content_type="application/octet-stream",
        )

    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        if not self.bucket:
            LOGGER.warning("MODELS_BUCKET not configured; load_model_csv(S3) → None")
            return None
        key = f"{self.prefix}{filename}"
        try:
            # load_bucket_object for CSV is expected to return a DataFrame
            return load_bucket_object(self.bucket, key)
        except ClientError as e:
            err_code = (e.response.get("Error") or {}).get("Code")
            if err_code in ("NoSuchKey", "NoSuchBucket"):
                LOGGER.info("load_model_csv(S3) missing: s3://%s/%s", self.bucket, key)
                return None
            LOGGER.exception(
                "Error loading model CSV s3://%s/%s (code=%s): %s",
                self.bucket,
                key,
                err_code,
                e,
            )
            raise e
        except Exception as ex:
            LOGGER.exception(
                "Error loading model CSV s3://%s/%s: %s", self.bucket, key, ex
            )
            raise ex

    def load_json(self, filename: str) -> Optional[dict]:
        if not self.bucket:
            LOGGER.warning("MODELS_BUCKET not configured; load_model_json(S3) → None")
            return None
        key = f"{self.prefix}{filename}"
        try:
            return load_bucket_object(self.bucket, key)
        except ClientError as e:
            err_code = (e.response.get("Error") or {}).get("Code")
            if err_code in ("NoSuchKey", "NoSuchBucket"):
                LOGGER.info("load_model_json(S3) missing: s3://%s/%s", self.bucket, key)
                return None
            LOGGER.exception(
                "Error loading model JSON s3://%s/%s (code=%s): %s",
                self.bucket,
                key,
                err_code,
                e,
            )
            raise e
        except Exception as ex:
            LOGGER.exception(
                "Error loading model JSON s3://%s/%s: %s", self.bucket, key, ex
            )
            raise ex

    def run_exists(self) -> bool:
        if not self.bucket:
            LOGGER.warning("MODELS_BUCKET is not configured; model_run_exists(S3) → False")
            return False
        keys: List[str] = list_bucket_objects(self.bucket, prefix=self.prefix)
        exists = len(keys) > 0
        LOGGER.info(
            f"model_run_exists(S3) run_id={self.run_id} → {exists} (keys={len(keys)})"
        )
        return exists

    def uri(self) -> str:
        return formulate_s3_uri(self.bucket, self.prefix)


def _get_model_store(run_id: str) -> _BaseModelStore:
    """
    Factory that returns the appropriate model store implementation for the current backend.
    """
    if is_s3():
        return _S3ModelStore(run_id)
    return _LocalModelStore(run_id)


# -------------------------------------------------------------------------
# Model artifacts - common save logic
# -------------------------------------------------------------------------
def _save_model_artifacts_to_store(
        store: _BaseModelStore,
        run_id: str,
        base_out: Dict[str, Any],
        stacked: Dict[str, Any],
        comb_avg: Dict[str, Any],
        comb_wgt_inv_rmse: Dict[str, Any],
        params_map: Dict[str, Any],
        y_true,
        y_pred,
        pred_src: str,
        model_name: str,
        x_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        x_sample: pd.DataFrame | None = None,
) -> str:
    """
    Backend-agnostic logic for saving all model artifacts for a run.
    The only difference between LOCAL and S3 is implemented in the ModelStore.
    """

    # --- per-model metrics ---
    if "per_model_metrics" in base_out:
        df_metrics = pd.DataFrame(base_out["per_model_metrics"])
        store.save_csv("per_model_metrics.csv", df_metrics)

    # --- ensembles (drop raw preds for JSON cleanliness) ---
    avg_to_save = {k: v for k, v in (comb_avg or {}).items() if k != "pred"}
    wgt_to_save = {k: v for k, v in (comb_wgt_inv_rmse or {}).items() if k != "pred"}

    store.save_json("ensemble_avg.json", avg_to_save)
    store.save_json("ensemble_weighted.json", wgt_to_save)

    # --- params map ---
    store.save_json("params_map.json", params_map or {})

    # --- predictions (y_true / y_pred from chosen prediction source) ---
    df_pred = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "pred_source": pred_src,
        }
    )
    store.save_csv("predictions.csv", df_pred)

    # --- stacked hybrid model (primary deployable) ---
    if stacked and isinstance(stacked, dict) and "model" in stacked:
        try:
            store.save_joblib(f"{model_name}.joblib", stacked["model"])
        except Exception as ex:  # noqa: BLE001
            LOGGER.warning(
                "Could not save stacked ensemble model for run %s to %s: %s",
                run_id,
                store.uri(),
                ex,
            )

        hybrid_meta = {
            "model_name": model_name,
            "kind": "stacked",
            "metrics": stacked.get("metrics", {}),
            "base_model_names": stacked.get("base_model_names", []),
        }
        store.save_json("ensemble_stacked.json", hybrid_meta)

    # --- base estimators (diagnostics / comparison only) ---
    models = base_out.get("models") or base_out.get("fitted")
    if isinstance(models, dict):
        for name, est in models.items():
            try:
                store.save_joblib(f"{name}.joblib", est)
            except Exception as ex:  # noqa: BLE001
                LOGGER.warning(
                    "Could not save base estimator '%s' for run %s to %s: %s",
                    name,
                    run_id,
                    store.uri(),
                    ex,
                )

    # --- explainability parameters ---
    explain_params: Dict[str, Any] = {}

    if x_valid is not None:
        explain_params["X_valid"] = {
            "columns": list(x_valid.columns),
            "data": x_valid.to_numpy().tolist(),
        }

    if y_valid is not None:
        # ensure plain Python list
        y_arr = np.asarray(y_valid).ravel().tolist()
        explain_params["y_valid"] = y_arr

    if x_sample is not None:
        explain_params["X_sample"] = {
            "columns": list(x_sample.columns),
            "data": x_sample.to_numpy().tolist(),
        }

    # Only write the file if we have something to save
    if explain_params:
        store.save_json("explain_params.json", explain_params)

    LOGGER.info(
        "Saved model artifacts (%s) → %s",
        "S3" if is_s3() else "LOCAL",
        store.uri(),
    )
    return store.uri()


# -------------------------------------------------------------------------
# Model artifacts - public persist API
# -------------------------------------------------------------------------
def save_model_artifacts(
        run_id: str,
        base_out: Dict[str, Any],
        stacked: Dict[str, Any],
        comb_avg: Dict[str, Any],
        comb_wgt_inv_rmse: Dict[str, Any],
        params_map: Dict[str, Any],
        y_true,
        y_pred,
        pred_src: str,
        model_name: str,
        x_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        x_sample: pd.DataFrame | None = None,
) -> str:
    """
    Persist model artifacts for a run, abstracting over LOCAL vs S3.

    Args:
        run_id:            Unique pipeline run identifier.
        base_out:          Dict returned by train_models_parallel (per_model_metrics, models, valid_preds, etc.).
        stacked:           Dict returned by build_stacked_ensemble (must contain "model", "metrics", "base_model_names").
        comb_avg:          Dict from combine_average (may include "pred" and "metrics").
        comb_wgt_inv_rmse: Dict from combine_weighted_inverse_rmse.
        params_map:        Hyperparameter mapping for trained models.
        y_true:            Validation / holdout true values.
        y_pred:            Validation / holdout predictions from the chosen source (typically stacked ensemble).
        pred_src:          String describing prediction source ("stacked_ensemble", etc.).
        model_name:        File/registry name for the hybrid model artifact (without extension).

    Returns:
        LOCAL → filesystem directory as str
        S3    → s3://BUCKET/run_id/ prefix as a pseudo 'dir'
    """
    store = _get_model_store(run_id)
    out_path = _save_model_artifacts_to_store(
        store=store,
        run_id=run_id,
        base_out=base_out,
        stacked=stacked,
        comb_avg=comb_avg,
        comb_wgt_inv_rmse=comb_wgt_inv_rmse,
        params_map=params_map,
        y_true=y_true,
        y_pred=y_pred,
        pred_src=pred_src,
        model_name=model_name,
        x_valid=x_valid,
        y_valid=y_valid,
        x_sample=x_sample,
    )
    return out_path


# -------------------------------------------------------------------------
# Model artifacts - public load / existence API
# -------------------------------------------------------------------------
def model_run_exists(run_id: str) -> bool:
    """
    True if there is *any* model artifact for this run_id,
    abstracting over LOCAL vs S3.
    """
    try:
        store = _get_model_store(run_id)
    except ValueError:
        # MODELS_BUCKET misconfigured for S3
        LOGGER.warning("Could not create model store for run_id=%s", run_id)
        return False
    return store.run_exists()


def load_model_csv(run_id: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV model artifact (predictions, per-model metrics) for a run_id.
    Returns None if the file doesn't exist.
    """
    store = _get_model_store(run_id)
    return store.load_csv(filename)


def load_model_json(run_id: str, filename: str) -> Optional[dict]:
    """
    Load a JSON model artifact (ensemble summaries, params map) for a run_id.
    Returns None if the file doesn't exist.
    """
    store = _get_model_store(run_id)
    return store.load_json(filename)


def load_stacked_model_for_run(run_id: str, model_name: str):
    """
    Load the stacked model <model_name>.joblib for a given run_id.

    Matches the save path used in _save_model_artifacts_to_store:
        store.save_joblib(f"{model_name}.joblib", stacked["model"])

    So the layout is:
        LOCAL: <MODELS_DIR>/<run_id>/<model_name>.joblib
        S3:    s3://<MODELS_BUCKET>/<run_id>/<model_name>.joblib
    """
    if not model_name:
        return None

    LOGGER.info(f"Loading stacked model for run_id={run_id}")

    run_id = (run_id or "").strip("/")
    filename = f"{model_name}.joblib"

    # ----------------------------
    # S3 BACKEND
    # ----------------------------
    if is_s3():
        bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
        if not bucket:
            LOGGER.warning("MODELS_BUCKET not defined; cannot load stacked model.")
            return None

        key = f"{run_id}/{filename}"
        LOGGER.info(f"Model S3 Key = {key}")
        try:
            obj = load_bucket_object(bucket, key)
        except Exception as ex:
            LOGGER.exception(f"Stacked model not found in S3: s3://{bucket}/{key}")
            return None

        # load_bucket_object may return raw bytes, Body.read(), or dict
        if isinstance(obj, (bytes, bytearray)):
            data = obj
        elif hasattr(obj, "read"):
            data = obj.read()
        elif isinstance(obj, dict) and "Body" in obj:
            data = obj["Body"].read()
        else:
            # Fallback: attempt to treat obj as bytes
            try:
                data = bytes(obj)
            except Exception:
                LOGGER.error("Unsupported S3 object format for stacked model: %s", type(obj))
                return None

        buf = io.BytesIO(data)
        try:
            return joblib.load(buf)
        except Exception as ex:
            LOGGER.exception("Failed to load stacked model joblib from S3 key %s: %s", key, ex)
            return None

    # ----------------------------
    # LOCAL BACKEND
    # ----------------------------
    models_root = Path(getattr(SETTINGS, "MODELS_DIR", "models"))
    path = models_root / run_id / filename

    if not path.exists():
        LOGGER.info("No stacked model found locally at %s", path)
        return None

    try:
        return joblib.load(path)
    except Exception as ex:
        LOGGER.exception("Failed to load stacked model at %s: %s", path, ex)
        return None


# -------------------------------------------------------------------------
# Model registry (global index of trained hybrid models)
# -------------------------------------------------------------------------

REGISTRY_FILENAME = "model_registry.json"


def _registry_path_local() -> Path:
    """
    Local filesystem path to the model registry index.
    """
    models_root = Path(getattr(SETTINGS, "MODELS_DIR", "models"))
    return models_root / REGISTRY_FILENAME


def _registry_bucket_key() -> tuple[Optional[str], str]:
    """
    (bucket, key) for the registry file in S3.
    """
    bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
    key = REGISTRY_FILENAME
    return bucket, key


def load_model_registry() -> list[dict]:
    """
    Load the global model registry index.

    LOCAL: <MODELS_DIR>/model_registry.json
    S3:    s3://<MODELS_BUCKET>/model_registry.json
    """
    if is_s3():
        bucket, key = _registry_bucket_key()
        if not bucket:
            LOGGER.warning("MODELS_BUCKET not configured; load_model_registry → []")
            return []

        try:
            data = load_bucket_object(bucket, key)
            # We expect the registry to be a list[dict]
            return data if isinstance(data, list) else []
        except ClientError as e:
            # Handle missing key/bucket as "registry not found"
            err_code = (e.response.get("Error") or {}).get("Code")
            if err_code in ("NoSuchKey", "NoSuchBucket"):
                LOGGER.info(
                    "Model registry not found in S3 (%s/%s, code=%s); returning empty list",
                    bucket,
                    key,
                    err_code,
                )
                return []
            # Anything else is a real error
            LOGGER.exception("Error loading model registry from S3: %s", e)
            raise e
        except Exception as ex:
            LOGGER.exception(f"Error loading model registry from S3: {ex}")
            raise
    else:
        path = _registry_path_local()
        if not path.exists():
            LOGGER.info("Model registry not found locally; returning empty list")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def save_model_registry(entries: list[dict]) -> None:
    """
    Persist the global model registry index.
    """
    if is_s3():
        bucket, key = _registry_bucket_key()
        if not bucket:
            LOGGER.warning("MODELS_BUCKET not configured; save_model_registry → no-op")
            return
        write_bucket_object(
            bucket,
            key,
            json.dumps(entries, indent=2).encode("utf-8"),
            content_type="application/json",
        )
    else:
        path = _registry_path_local()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)


def register_model_in_registry(
        run_id: str,
        model_name: str,
        status: str = "candidate",
        created_by: Optional[str] = None,
) -> None:
    """
    Register a hybrid model for a given run in the global registry.

    - `run_id` is the pipeline run identifier (same as used in MODELS_DIR / S3 prefix).
    - `model_name` is the SAME name used for the saved hybrid model artifact,
      e.g. 'ppe_model_<run_id>'.
    - `status` is typically 'candidate' (later you can promote to 'deployed').
    - `created_by` is a free-form username or id stored for audit.

    This function:
      1. Loads stacked ensemble for the run to capture metrics + base_model_names.
      2. Removes any existing entry with the same (run_id, model_name).
      3. Appends a new entry and writes the registry back.
    """
    # Pull metrics + base model names from per-run meta
    ensemble_meta_stacked = load_model_json(run_id, "ensemble_stacked.json") or {}
    metrics = ensemble_meta_stacked.get("metrics", {})
    base_model_names = ensemble_meta_stacked.get("base_model_names", [])

    registry = load_model_registry()

    # Remove any existing entry for same run + model_name
    registry = [
        entry
        for entry in registry
        if not (entry.get("run_id") == run_id and entry.get("model_name") == model_name)
    ]

    # Append new entry
    registry.append(
        {
            "run_id": run_id,
            "model_name": model_name,
            "status": status,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "created_by": created_by or "unknown",
            "metrics": metrics,
            "base_model_names": base_model_names,
        }
    )

    save_model_registry(registry)
    LOGGER.info(
        "Registered model in registry: run_id=%s, model_name=%s, status=%s",
        run_id,
        model_name,
        status,
    )
