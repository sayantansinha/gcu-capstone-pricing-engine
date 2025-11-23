from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Dict, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.config.env_loader import SETTINGS
from src.ui.common import store_last_model_info_in_session, extract_last_trained_models, \
    store_last_run_model_dir_in_session
from src.ui.pipeline.pipeline_flow import render_pipeline_flow
from src.ui.pipeline.steps import features, source_data_stager
from src.ui.pipeline.steps.modeling import render as render_models
from src.ui.pipeline.steps.cleaning import render_cleaning_section
from src.ui.pipeline.steps.display_data import render_display_section
from src.ui.pipeline.steps.exploration import render_exploration_section
from src.ui.pipeline.steps.reporting import render as render_reports
from src.ui.pipeline.steps.visual_tools import render as render_visuals
from src.utils.data_io_utils import latest_file_under_directory, load_processed
from src.utils.model_io_utils import model_run_exists, load_model_csv, load_model_json
from src.utils.log_utils import get_logger

LOGGER = get_logger("pipeline_hub")


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
      → returns full local paths as strings

    S3:
      under_dir = Path(run_id) → becomes "<run_id>" prefix inside PROCESSED_BUCKET
      → returns s3:// URIs as strings
    """
    if SETTINGS.IO_BACKEND == "S3":
        # For S3, under_dir is used as a *prefix*, not a real filesystem path.
        under_dir = Path(run_id)
    else:
        # For local, we actually need the concrete directory.
        under_dir = Path(SETTINGS.PROCESSED_DIR) / run_id

    fm_raw = latest_file_under_directory("feature_master_", under_dir, exclusion="cleaned")
    fm_clean = latest_file_under_directory("feature_master_cleaned_", under_dir)

    LOGGER.info(f"Probed feature master artifacts :: raw [{fm_raw}], clean [{fm_clean}]")
    return fm_raw, fm_clean


def _probe_model_artifacts(run_id: str) -> bool:
    """
    Backend-agnostic check: does this run_id have any model artifacts?
    Delegates to data_io_utils.model_run_exists(). \
    If it exists then activate model
    """
    has_model = model_run_exists(run_id)
    if has_model:
        _activate_model(run_id)

    return has_model


def _activate_feature_master(run_id: str, fm_raw: Optional[Path], fm_clean: Optional[Path]):
    """
    Load raw & cleaned Feature Master into session state.

    fm_* values are strings which may be:
      - local paths (LOCAL backend)
      - s3:// URIs (S3 backend)

    We map them back to the saved naming convention and use load_processed()
    so all backend-specific IO goes through data_io_utils.
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

    All IO goes through data_io_utils/model_io_utils, so this function is backend-agnostic.
    """

    # ---------- Predictions (required) ----------
    pred_df = load_model_csv(run_id, "predictions.csv")
    if pred_df is None:
        LOGGER.info(f"No predictions.csv found for run_id={run_id}")
        return False

    try:
        y_true = pred_df["y_true"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()
        if "pred_source" in pred_df.columns and len(pred_df):
            pred_src = str(pred_df["pred_source"].iloc[0])
        else:
            pred_src = "unknown"

        # ---------- Ensemble summaries (JSON, optional) ----------
        ensemble_avg = load_model_json(run_id, "ensemble_avg.json") or {}
        ensemble_wgt = load_model_json(run_id, "ensemble_weighted.json") or {}

        # ---------- Stacked ensemble meta ----------
        hybrid_meta = load_model_json(run_id, "ensemble_stacked.json") or {}
        stacked_metrics = hybrid_meta.get("metrics") or {}
        model_name = hybrid_meta.get("model_name") or f"ppe_model_{run_id}"

        # We don't have stacked predictions when reloading; only metrics.
        stacked = {"metrics": stacked_metrics}

        # ---------- Per-model metrics (CSV, optional) ----------
        pm_metrics_df = load_model_csv(run_id, "per_model_metrics.csv")
        if pm_metrics_df is not None:
            trained_models = pm_metrics_df["model"].dropna().astype(str).tolist()
            per_model_metrics = pm_metrics_df.to_dict(orient="records")
        else:
            trained_models = []
            per_model_metrics = []

        # ---------- Hyperparameters map (JSON, optional) ----------
        params_map = load_model_json(run_id, "params_map.json") or {}

        # ---------- Store in session (keep signature consistent with modeling.py) ----------
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
            model_name
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


def _render_current_artifacts(
        fm_raw_path: Path | None,
        fm_clean_path: Path | None,
        has_model: bool
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
        has_model: bool
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
        # First render current artifacts
        _render_current_artifacts(fm_raw_path, fm_clean_path, has_model)

        # Then Render pipeline flow diagram
        render_pipeline_flow(ctx)


def render():
    run_id = st.session_state.get("run_id")
    LOGGER.info(f"Rendering pipeline_hub for run_id [{run_id}]")
    if not isinstance(run_id, str) or not run_id.strip():
        st.error("run_id is not initialized. Ensure app.py sets it before loading the hub.")
        return

    st.header(f"Pipeline – {run_id}")

    # Probe and reload feature master artifacts
    fm_raw_path, fm_clean_path = _probe_feature_master_artifacts(run_id)

    # Probe model artifacts
    has_model = _probe_model_artifacts(run_id)

    # Status strip for clarity when resuming
    # with st.container(border=True):
    # Reserve a fixed slot to render
    pipeline_flow_slot = st.empty()

    # Render Pipeline state
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
