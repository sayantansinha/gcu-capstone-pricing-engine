from __future__ import annotations

import contextlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Set

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.config.env_loader import SETTINGS
from src.ui.common import (
    store_last_model_info_in_session,
    extract_last_trained_models,
    store_last_run_model_dir_in_session,
)
from src.ui.pipeline.pipeline_flow import render_pipeline_flow
from src.ui.pipeline.steps import features, source_data_stager
from src.ui.pipeline.steps.modeling import render as render_models
from src.ui.pipeline.steps.cleaning import render_cleaning_section
from src.ui.pipeline.steps.display_data import render_display_section
from src.ui.pipeline.steps.exploration import render_exploration_section
from src.ui.pipeline.steps.reporting import render as render_reports
from src.ui.pipeline.steps.visual_tools import render as render_visuals
from src.utils.data_io_utils import latest_file_under_directory, load_processed
from src.utils.model_io_utils import model_run_exists, load_model_csv, load_model_json, load_model_registry
from src.utils.log_utils import get_logger
from src.utils.s3_utils import list_bucket_objects

LOGGER = get_logger("ui_pipeline_hub")


# -------------------------------------------------------------------
# Run management helpers (for dropdown + New Pipeline)
# -------------------------------------------------------------------
def _new_run_id() -> str:
    """Generate a new run id: YYYYMMDD-HHMMSS_XXXXXXXX."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _ensure_run_dirs(run_id: str) -> None:
    """
    For LOCAL backend, ensure run-specific directories exist.
    For S3 backend, directories are virtual (prefix-based), so nothing is needed.
    """
    if SETTINGS.IO_BACKEND == "S3":
        return

    for root in (
            Path(SETTINGS.RAW_DIR),
            Path(SETTINGS.PROCESSED_DIR),
            Path(SETTINGS.FIGURES_DIR),
            Path(SETTINGS.PROFILES_DIR),
            Path(SETTINGS.MODELS_DIR),
            Path(SETTINGS.REPORTS_DIR),
    ):
        (root / run_id).mkdir(parents=True, exist_ok=True)


def _list_run_ids() -> List[str]:
    """
    Discover existing run_ids from either LOCAL or S3.

    LOCAL:
      - directory names under RAW_DIR / PROCESSED_DIR / MODELS_DIR

    S3:
      - first path segment of keys in RAW_BUCKET / PROCESSED_BUCKET / MODELS_BUCKET
    """
    run_ids: Set[str] = set()

    if SETTINGS.IO_BACKEND == "S3":
        # RAW
        if SETTINGS.RAW_BUCKET:
            for key in list_bucket_objects(SETTINGS.RAW_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) >= 2:
                    run_ids.add(parts[0])
        # PROCESSED
        if SETTINGS.PROCESSED_BUCKET:
            for key in list_bucket_objects(SETTINGS.PROCESSED_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) >= 2:
                    run_ids.add(parts[0])
        # MODELS
        models_bucket = getattr(SETTINGS, "MODELS_BUCKET", None)
        if models_bucket:
            for key in list_bucket_objects(models_bucket, prefix=""):
                parts = key.split("/")
                if len(parts) >= 2:
                    run_ids.add(parts[0])
    else:
        # LOCAL
        for root in (
                Path(SETTINGS.RAW_DIR),
                Path(SETTINGS.PROCESSED_DIR),
                Path(getattr(SETTINGS, "MODELS_DIR", "")),
        ):
            if not root or not root.exists():
                continue
            for run_dir in root.iterdir():
                if run_dir.is_dir():
                    run_ids.add(run_dir.name)

    # Sort newest first assuming run_ids start with timestamp
    sorted_ids = sorted(run_ids, reverse=True)
    LOGGER.info("Discovered pipeline runs: %s", sorted_ids)
    return sorted_ids


def _init_run(run_id: str, is_new: bool = False) -> None:
    """
    Initialize session state for a selected or newly created run.
    """
    st.session_state["run_id"] = run_id
    # Clear transient state so we don't carry over old artifacts
    st.session_state.pop("staged_raw", None)
    st.session_state.pop("_raw_preview", None)
    st.session_state.pop("last_model", None)
    st.session_state.pop("last_model_run_dir", None)
    if is_new:
        LOGGER.info("Initialized NEW pipeline run [%s]", run_id)
    else:
        LOGGER.info("Activated EXISTING pipeline run [%s]", run_id)

    _ensure_run_dirs(run_id)


# -------------------------------------------------------------------
# Existing artifact / pipeline helpers
# -------------------------------------------------------------------
def _artifacts(run_id: str) -> Dict[str, Optional[str]]:
    """
    Lightweight snapshot of artifacts for the status strip.

    fm_raw / fm_clean are strings (local paths or S3 URIs),
    model is a simple boolean flag (plus in-session fallback),
    so this stays backend-agnostic.
    """
    fm_raw, fm_clean = _probe_feature_master_artifacts(run_id)

    if "last_model" in st.session_state:
        # in-session model, freshly trained
        has_model = True
    else:
        has_model = _probe_model_artifacts(run_id)

    return {
        "fm_raw": fm_raw,
        "fm_clean": fm_clean,
        "model": has_model,
    }


def _probe_feature_master_artifacts(run_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate latest raw & cleaned Feature Master for a given run.

    LOCAL:
      under_dir = PROCESSED_DIR / run_id

    S3:
      under_dir = Path(run_id) → becomes "<run_id>" prefix inside PROCESSED_BUCKET
    """
    if SETTINGS.IO_BACKEND == "S3":
        under_dir = Path(run_id)  # used as prefix
    else:
        under_dir = Path(SETTINGS.PROCESSED_DIR) / run_id

    fm_raw = latest_file_under_directory("feature_master_", under_dir, exclusion="cleaned")
    fm_clean = latest_file_under_directory("feature_master_cleaned_", under_dir)

    LOGGER.info("Probed feature master artifacts :: raw [%s], clean [%s]", fm_raw, fm_clean)
    return fm_raw, fm_clean


def _probe_model_artifacts(run_id: str) -> bool:
    """
    Backend-agnostic check: does this run_id have any model artifacts?
    Delegates to model_run_exists().
    If it exists then activate model.
    """
    has_model = model_run_exists(run_id)
    if has_model:
        _activate_model(run_id)

    return has_model


def _activate_feature_master(run_id: str, fm_raw: Optional[Path], fm_clean: Optional[Path]):
    """
    Load raw & cleaned Feature Master into session state.
    """
    # Raw Feature Master
    if fm_raw:
        try:
            raw_name = Path(fm_raw).stem  # feature_master_YYYY...
            df = load_processed(raw_name, base_dir=run_id)
            st.session_state["last_feature_master_path"] = fm_raw
            st.session_state["df"] = df
        except Exception as e:
            label = Path(fm_raw).name
            st.warning(f"Could not load feature master {label}: {e}")
    else:
        LOGGER.info("No raw feature master found, use Build Feature Master first.")

    # Cleaned Feature Master
    if fm_clean:
        try:
            clean_name = Path(fm_clean).stem  # feature_master_cleaned_YYYY...
            df = load_processed(clean_name, base_dir=run_id)
            st.session_state["last_cleaned_feature_master_path"] = fm_clean
            st.session_state["cleaned_df"] = df
            st.session_state["preprocessing_performed"] = True
        except Exception as e:
            label = Path(fm_clean).name
            st.warning(f"Could not load clean feature master {label}: {e}")
    else:
        st.session_state["preprocessing_performed"] = False
        LOGGER.info("No clean feature master found, save a cleaned feature master first.")


def _activate_model(run_id: str) -> bool:
    """
    Load model artifacts (predictions, ensemble summaries, per-model metrics, params)
    for a given run_id from either LOCAL or S3, and stash them into session_state.
    """
    pred_df = load_model_csv(run_id, "predictions.csv")
    if pred_df is None:
        LOGGER.info("No predictions.csv found for run_id=%s", run_id)
        return False

    try:
        y_true = pred_df["y_true"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()
        if "pred_source" in pred_df.columns and len(pred_df):
            pred_src = str(pred_df["pred_source"].iloc[0])
        else:
            pred_src = "unknown"

        ensemble_avg = load_model_json(run_id, "ensemble_avg.json") or {}
        ensemble_wgt = load_model_json(run_id, "ensemble_weighted.json") or {}
        hybrid_meta = load_model_json(run_id, "ensemble_stacked.json") or {}
        stacked_metrics = hybrid_meta.get("metrics") or {}
        model_name = hybrid_meta.get("model_name") or f"ppe_model_{run_id}"

        stacked = {"metrics": stacked_metrics}

        pm_metrics_df = load_model_csv(run_id, "per_model_metrics.csv")
        if pm_metrics_df is not None:
            trained_models = pm_metrics_df["model"].dropna().astype(str).tolist()
            per_model_metrics = pm_metrics_df.to_dict(orient="records")
        else:
            trained_models = []
            per_model_metrics = []

        params_map = load_model_json(run_id, "params_map.json") or {}

        base_for_session = {"per_model_metrics": per_model_metrics}

        store_last_model_info_in_session(
            base_for_session,
            stacked,
            ensemble_avg,
            ensemble_wgt,
            y_true,
            y_pred,
            pred_src,
            params_map,
            trained_models,
            model_name,
        )

        store_last_run_model_dir_in_session(run_id)

        st.session_state["model_trained"] = True
        return True

    except Exception as e:
        st.error("Could not activate model run, check logs for details")
        LOGGER.exception("Error in _activate_model", exc_info=e)
        return False


@contextlib.contextmanager
def _suppress_child_section_panels():
    prev = st.session_state.get("_suppress_section_panel", False)
    st.session_state["_suppress_section_panel"] = True
    try:
        yield
    finally:
        st.session_state["_suppress_section_panel"] = prev


def _compute_run_stage_for_chip(run_id: str) -> tuple[str, str]:
    """
    Compute a human-readable stage for the chip at the top toolbar
    and return (label, css_class_suffix).

    Order of precedence (most advanced to least):
      - Model registered
      - Model trained
      - Cleaned
      - Features built
      - Initiated
    """
    # Probe feature master artifacts
    fm_raw_path, fm_clean_path = _probe_feature_master_artifacts(run_id)

    # Check if any model artifacts exist (without forcing activation)
    has_model = model_run_exists(run_id) or bool(st.session_state.get("last_model"))

    # Check model registry for this run_id
    registered_run_ids = set()
    try:
        registry_entries = load_model_registry()
        registered_run_ids = {
            entry.get("run_id")
            for entry in registry_entries
            if entry.get("run_id") is not None
        }
    except Exception as ex:
        LOGGER.warning(
            "Could not load model registry in _compute_run_stage_for_chip: %s", ex
        )

    # Order of precedence, same as original infer_stage
    if run_id in registered_run_ids:
        return "Model registered", "stage-registered"
    if has_model:
        return "Model trained", "stage-model"
    if fm_clean_path is not None:
        return "Cleaned", "stage-clean"
    if fm_raw_path is not None:
        return "Features built", "stage-features"

    return "Initiated", "stage-new"


def _format_run_timestamp_from_id(run_id: str) -> str:
    """
    Format the timestamp encoded in run_id as a human-readable string.

    Example run_id: 20251122-192908_94605c58
    → 'Nov 22, 2025 07:29 PM'
    """
    try:
        head = run_id.split("_")[0]
        date_part, time_part = head.split("-") if "-" in head else head.split("_")
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        return dt.strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return "N/A"


def _render_current_artifacts(
        fm_raw_path: Path | None,
        fm_clean_path: Path | None,
        has_model: bool,
):
    fm_raw_label = fm_raw_path if fm_raw_path else "N/A"
    fm_clean_label = fm_clean_path if fm_clean_path else "N/A"
    last_trained_models = extract_last_trained_models(True) if has_model else "N/A"
    st.markdown(
        f"> **Current artifacts** — Feature Master (raw): **{fm_raw_label}** "
        f"| Feature Master (cleaned): **{fm_clean_label}** | Base Model(s): **{last_trained_models}**"
    )


def _render_pipeline_state(
        pipeline_flow_slot: DeltaGenerator,
        fm_raw_path: Path | None,
        fm_clean_path: Path | None,
        has_model: bool,
):
    """
    Render the pipeline state
    """
    ctx = {
        "files_staged": st.session_state.get("staged_files_count", 0) > 0,
        "feature_master_exists": st.session_state.get("last_feature_master_path") is not None,
        "data_displayed": st.session_state.get("data_displayed"),
        "eda_performed": st.session_state.get("eda_performed"),
        "preprocessing_performed": st.session_state.get("preprocessing_performed"),
        "model_trained": st.session_state.get("model_trained"),
        "report_generated": st.session_state.get("report_generated"),
    }

    with pipeline_flow_slot.container(border=True):
        _render_current_artifacts(fm_raw_path, fm_clean_path, has_model)
        render_pipeline_flow(ctx)


# -------------------------------------------------------------------
# Main render
# -------------------------------------------------------------------
def render():
    # Discover existing runs for dropdown
    run_ids = _list_run_ids()
    current_run_id = st.session_state.get("run_id")

    st.header("Pipeline")

    # -----------------------------------------------------------
    # Toolbar: dropdown + icon button in one row
    # -----------------------------------------------------------
    st.markdown("<div class='pipeline-toolbar'>", unsafe_allow_html=True)
    col_select, col_btn = st.columns([4, 1])

    # ---- dropdown ----
    with col_select:
        st.markdown("<div class='toolbar-new-pipeline'>", unsafe_allow_html=True)
        if run_ids:
            default_index = run_ids.index(current_run_id) if current_run_id in run_ids else 0
            selected_run_id = st.selectbox(
                "Pipeline Runs",
                options=run_ids,
                index=default_index,
                key="pipeline_run_select",
                label_visibility="collapsed",
            )

            if selected_run_id != current_run_id:
                _init_run(selected_run_id, is_new=False)
                st.rerun()
        else:
            st.info("No pipeline runs yet. Use the ➕ button to create one.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- icon button for new pipeline ----
    with col_btn:
        st.markdown("<div class='toolbar-new-pipeline'>", unsafe_allow_html=True)
        if st.button("➕", key="btn_new_run",
                     help="Create new pipeline run",
                     use_container_width=True):
            new_id = _new_run_id()
            _init_run(new_id, is_new=True)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end .pipeline-toolbar

    # -----------------------------------------------------------
    # Active run + chip row
    # -----------------------------------------------------------
    run_id = st.session_state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        # No active run selected/created yet; nothing more to render.
        return

    stage_label, stage_class = _compute_run_stage_for_chip(run_id)
    last_update = _format_run_timestamp_from_id(run_id)

    st.markdown(
        f"""
        <div class="active-run-row">
          <span class="active-run-label">Active run:</span>
          <span class="active-run-id">{run_id}</span>
          <span class="run-stage-chip {stage_class}">{stage_label}</span>
          <span class="run-last-update">Last updated on {last_update}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    LOGGER.info("Rendering pipeline_hub for run_id [%s]", run_id)

    # Probe and reload feature master artifacts
    fm_raw_path, fm_clean_path = _probe_feature_master_artifacts(run_id)

    # Probe model artifacts
    has_model = _probe_model_artifacts(run_id)

    pipeline_flow_slot = st.empty()

    # Initial pipeline state
    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)

    # Stage Sources
    with st.expander("Data Staging", expanded=False):
        source_data_stager.render()

    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)

    # Build Feature
    with st.expander("Feature Master", expanded=False):
        features.render()

    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)

    # Probe feature master before data exploration
    fm_raw_path, fm_clean_path = _probe_feature_master_artifacts(run_id)
    _activate_feature_master(run_id, fm_raw_path, fm_clean_path)
    LOGGER.info("Feature master (raw) artifacts loaded and session states activated")

    # Display Data
    user_msg: str = "Build a Feature Master first."
    with st.expander("Display Data", expanded=False):
        if fm_raw_path:
            st.session_state["data_displayed"] = True
            with _suppress_child_section_panels():
                render_display_section()
        else:
            st.session_state["data_displayed"] = False
            st.info(user_msg)

    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)

    # Exploration (EDA)
    with st.expander("Exploration (EDA)", expanded=False):
        if fm_raw_path:
            st.session_state["eda_performed"] = True
            with _suppress_child_section_panels():
                render_exploration_section()
        else:
            st.session_state["eda_performed"] = False
            st.info(user_msg)

    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)

    # Cleaning & Preprocessing
    with st.expander("Preprocessing (and Cleaning)", expanded=False):
        if fm_raw_path:
            with _suppress_child_section_panels():
                render_cleaning_section()
        else:
            st.session_state["preprocessing_performed"] = False
            st.info(user_msg)

    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)

    # Re-probe after cleaning, only reading cleaned feature master
    _, fm_clean_path = _probe_feature_master_artifacts(run_id)
    _activate_feature_master(run_id, fm_raw_path, fm_clean_path)
    LOGGER.info("Feature master (clean) artifacts loaded and session states activated")

    # Modeling
    with st.expander("Modeling", expanded=False):
        if fm_clean_path:
            with _suppress_child_section_panels():
                render_models()
        else:
            st.session_state["model_trained"] = False
            st.info("Save a cleaned Feature Master to enable modeling.")

    # Re-probe after modeling and re-render flow diagram
    has_model = _probe_model_artifacts(run_id)
    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)

    # Visual Tools
    with st.expander("Visual Tools", expanded=False):
        if has_model:
            with _suppress_child_section_panels():
                render_visuals()
        else:
            st.info("Train a model to enable visual tools.")

    # Reporting
    with st.expander("Reporting", expanded=False):
        if has_model:
            with _suppress_child_section_panels():
                render_reports()
        else:
            st.session_state["report_generated"] = False
            st.info("Train a model to enable the report generator.")

    _render_pipeline_state(pipeline_flow_slot, fm_raw_path, fm_clean_path, has_model)
