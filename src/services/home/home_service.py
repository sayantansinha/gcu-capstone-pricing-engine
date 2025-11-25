from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from src.config.env_loader import SETTINGS
from src.utils.data_io_utils import RunInfo, is_s3
from src.utils.log_utils import get_logger
from src.utils.model_io_utils import (
    model_run_exists,
    load_model_json,
    load_model_registry,
)
from src.utils.s3_utils import list_bucket_objects

LOGGER = get_logger("home_service")


# --------------------------------------------------------------------
# Internal helpers: run discovery
# --------------------------------------------------------------------
def _list_model_run_ids() -> List[str]:
    """
    List all known run_ids based on the model artifact storage.

    LOCAL:
        - Uses <MODELS_DIR> and returns subdirectory names except the registry file.

    S3:
        - Uses MODELS_BUCKET and returns the first path segment
          before '/' as run_id.
    """
    if is_s3():
        bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
        if not bucket:
            LOGGER.warning("MODELS_BUCKET not configured; _list_model_run_ids → []")
            return []

        keys = list_bucket_objects(bucket, prefix="")
        run_ids = {
            key.split("/", 1)[0]
            for key in keys
            if "/" in key  # ignore any weird top-level files
        }
        out = sorted(run_ids)
        LOGGER.info("Discovered model runs (S3) → %s", out)
        return out

    # LOCAL
    models_root = Path(getattr(SETTINGS, "MODELS_DIR", "models"))
    if not models_root.exists():
        LOGGER.info("Models root not found (LOCAL): %s", models_root)
        return []

    run_ids: List[str] = []
    for entry in models_root.iterdir():
        # Skip the model registry JSON
        if entry.is_file() and entry.name == "model_registry.json":
            continue
        if entry.is_dir():
            run_ids.append(entry.name)

    out = sorted(run_ids)
    LOGGER.info("Discovered model runs (LOCAL) → %s", out)
    return out


def get_latest_run_id() -> Optional[str]:
    """
    Best-effort: returns the lexicographically latest run_id.
    Assumes run_ids contain a timestamp-like suffix so that
    newest runs sort last.
    """
    run_ids = _list_model_run_ids()
    if not run_ids:
        return None
    latest = sorted(run_ids)[-1]
    LOGGER.info("Latest run_id → %s", latest)
    return latest


# --------------------------------------------------------------------
# Pipeline stage status (RunInfo)
# --------------------------------------------------------------------
def _has_any_parquet_in_raw(run_id: str) -> bool:
    if is_s3():
        bucket = getattr(SETTINGS, "RAW_BUCKET", None)
        if not bucket:
            return False
        prefix = f"{run_id.strip('/')}/"
        keys = [
            k for k in list_bucket_objects(bucket, prefix=prefix)
            if k.endswith(".parquet")
        ]
        return len(keys) > 0

    raw_dir = Path(getattr(SETTINGS, "RAW_DIR", "data/raw")) / run_id
    if not raw_dir.exists():
        return False
    return any(p.is_file() and p.suffix == ".parquet" for p in raw_dir.iterdir())


def _feature_master_flags(run_id: str) -> Tuple[bool, bool]:
    """
    Returns (has_feature_master, has_feature_master_cleaned),
    based on PROCESSED storage for this run_id.

    We infer using filename conventions:
      - contains "feature_master" → base master
      - contains both "feature_master" and "clean" → cleaned
    """
    has_fm = False
    has_fm_clean = False

    if is_s3():
        bucket = getattr(SETTINGS, "PROCESSED_BUCKET", None)
        if not bucket:
            return False, False

        prefix = f"{run_id.strip('/')}/"
        keys = list_bucket_objects(bucket, prefix=prefix)

        for k in keys:
            name = k.split("/")[-1].lower()
            if "feature_master" in name:
                if "clean" in name:
                    has_fm_clean = True
                else:
                    has_fm = True
        return has_fm, has_fm_clean

    processed_root = Path(getattr(SETTINGS, "PROCESSED_DIR", "data/processed")) / run_id
    if not processed_root.exists():
        return False, False

    for p in processed_root.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if "feature_master" in name:
            if "clean" in name:
                has_fm_clean = True
            else:
                has_fm = True

    return has_fm, has_fm_clean


def _has_registered_model(run_id: str) -> bool:
    """
    Returns True if the global registry contains an entry for this run_id.
    """
    registry = load_model_registry()
    for entry in registry:
        if entry.get("run_id") == run_id:
            return True
    return False


def get_run_info(run_id: str) -> RunInfo:
    """
    Build a RunInfo instance for a given run_id using the current IO backend.
    """
    has_raw = _has_any_parquet_in_raw(run_id)
    has_fm, has_fm_clean = _feature_master_flags(run_id)
    has_model = model_run_exists(run_id)
    has_registered = _has_registered_model(run_id)

    info = RunInfo(
        run_id=run_id,
        has_raw=has_raw,
        has_feature_master=has_fm,
        has_feature_master_cleaned=has_fm_clean,
        has_model=has_model,
        has_registered_model=has_registered,
    )
    LOGGER.info("RunInfo(%s) → %s", run_id, asdict(info))
    return info


def get_latest_run_info() -> Optional[RunInfo]:
    """
    Convenience: RunInfo for the most recent run, or None.
    """
    latest = get_latest_run_id()
    if not latest:
        return None
    return get_run_info(latest)


# --------------------------------------------------------------------
# Model insights (metrics + BP test)
# --------------------------------------------------------------------
def _extract_metric(metrics: Dict[str, Any], *names: str) -> Optional[float]:
    """
    Case-insensitive lookup for metric keys.
    """
    lowered = {k.lower(): v for k, v in metrics.items()}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def _extract_bp_summary(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Heuristics to extract Breusch–Pagan test results from metrics.

    We support:
      - metrics["bp_test"] or metrics["bp"] as a dict
      - individual keys like "bp_stat", "bp_statistic", "bp_pvalue", "bp_p"
    """
    if not metrics:
        return None

    # Nested dicts
    for key in ("bp_test", "bp"):
        if isinstance(metrics.get(key), dict):
            return metrics[key]

    # Flat keys
    stat = metrics.get("bp_stat") or metrics.get("bp_statistic")
    pval = metrics.get("bp_pvalue") or metrics.get("bp_p")
    if stat is None and pval is None:
        return None

    return {
        "stat": stat,
        "pvalue": pval,
    }


def get_latest_model_insights(run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with:
      {
        "run_id": str,
        "model_name": str,
        "rmse": float | None,
        "mae": float | None,
        "r2": float | None,
        "metrics": dict,
        "bp_summary": dict | None,
    }
    or None if no suitable run/model is found.
    """
    if not run_id:
        run_id = get_latest_run_id()
        if not run_id:
            LOGGER.info("No runs found; get_latest_model_insights → None")
            return None

    meta = load_model_json(run_id, "ensemble_stacked.json") or {}
    metrics: Dict[str, Any] = meta.get("metrics", {}) or {}

    rmse = _extract_metric(metrics, "rmse", "root_mean_squared_error")
    mae = _extract_metric(metrics, "mae", "mean_absolute_error")
    r2 = _extract_metric(metrics, "r2", "r_squared")
    bp_summary = _extract_bp_summary(metrics)

    model_name = meta.get("model_name") or "ensemble_stacked"

    insights = {
        "run_id": run_id,
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "metrics": metrics,
        "bp_summary": bp_summary,
    }
    LOGGER.info("Latest model insights for run %s → %s", run_id, insights)
    return insights


# --------------------------------------------------------------------
# System health
# --------------------------------------------------------------------
def _check_s3_connectivity() -> Tuple[bool, Optional[str]]:
    """
    Simple connectivity probe for S3-based backends.
    Tries to list objects from MODELS_BUCKET or RAW_BUCKET.
    """
    if not is_s3():
        return False, None

    bucket = getattr(SETTINGS, "MODELS_BUCKET", None) or getattr(
        SETTINGS, "RAW_BUCKET", None
    )
    if not bucket:
        return False, "No MODELS_BUCKET or RAW_BUCKET configured"

    try:
        # We don't care about actual contents, just that the call works
        list_bucket_objects(bucket, prefix="")
        return True, None
    except Exception as ex:  # pragma: no cover - best effort
        LOGGER.exception("S3 connectivity check failed: %s", ex)
        return False, str(ex)


def get_system_health() -> Dict[str, Any]:
    """
    Returns a dict summarizing system health for the Home dashboard.
    """
    env_name = getattr(SETTINGS, "ENV", None) or getattr(
        SETTINGS, "APP_ENV", None
    ) or "unknown"

    io_backend = getattr(SETTINGS, "IO_BACKEND", "LOCAL?")

    # Basic config presence checks
    required_attrs = ["RAW_DIR", "PROCESSED_DIR", "MODELS_DIR"]
    missing_config = [
        attr for attr in required_attrs if not hasattr(SETTINGS, attr)
    ]

    s3_enabled, s3_error = _check_s3_connectivity()
    app_version = getattr(SETTINGS, "APP_VERSION", "N/A")

    health = {
        "environment": env_name,
        "io_backend": io_backend,
        "config_ok": len(missing_config) == 0,
        "missing_config": missing_config,
        "s3_enabled": s3_enabled,
        "s3_error": s3_error,
        "app_version": app_version,
    }
    LOGGER.info("System health → %s", health)
    return health
