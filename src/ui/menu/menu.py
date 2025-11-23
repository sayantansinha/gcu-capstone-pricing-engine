from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st

from src.config.env_loader import SETTINGS
from src.ui.common import logo_path, APP_NAME
from src.utils.data_io_utils import RunInfo, _is_s3
from src.utils.log_utils import get_logger
from src.utils.model_io_utils import load_model_registry
from src.utils.s3_utils import list_bucket_objects

LOGGER = get_logger("ui_menu")


def _section_header():
    path = logo_path()
    if path:
        svg_text = path.read_text(encoding="utf-8")
        b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        st.sidebar.markdown(
            f"<div class='logo-container'>"
            f"<img src='data:image/svg+xml;base64,{b64}' />"
            f"<span class='app-name-text'>{APP_NAME}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.markdown("---")


# -------------------------------------------------------------------
# Run management helpers
# -------------------------------------------------------------------
def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _infer_stage(info: RunInfo) -> str:
    """
    Derive pipeline stage from run metadata flags.

    If a model for this run is already registered in the global model registry,
    we surface that as the most advanced status.
    """
    if info.has_registered_model:
        return "Model registered"
    if info.has_model:
        return "Model trained"
    if info.has_feature_master_cleaned:
        return "Cleaned"
    if info.has_feature_master:
        return "Features built"
    if info.has_raw:
        return "Raw staged"
    return "Initiated"


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _format_created_from_run_id(run_id: str) -> Optional[str]:
    try:
        head = run_id.split("_")[0]
        date_part, time_part = head.split("-") if "-" in head else head.split("_")
    except Exception:
        compact = run_id.replace("_", "").replace("-", "")
        if len(compact) >= 14:
            date_part, time_part = compact[:8], compact[8:14]
        else:
            return None
    try:
        yyyy, mm, dd = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8])
        HH, MM, SS = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6])
        ampm = "AM" if HH < 12 else "PM"
        hh12 = HH % 12 or 12
        return f"{_MONTHS[mm - 1]} {dd:02d}, {yyyy} {hh12:02d}:{MM:02d} {ampm}"
    except Exception:
        return None


def _ensure_run_dirs(run_id: str):
    for root in (
            Path(SETTINGS.RAW_DIR),
            Path(SETTINGS.PROCESSED_DIR),
            Path(SETTINGS.FIGURES_DIR),
            Path(SETTINGS.PROFILES_DIR),
            Path(SETTINGS.MODELS_DIR),
            Path(SETTINGS.REPORTS_DIR)
    ):
        (root / run_id).mkdir(parents=True, exist_ok=True)


def _list_runs() -> List[RunInfo]:
    """
    Collect run-level metadata for all runs, abstracting over LOCAL vs S3.

    - In LOCAL mode:
        RAW_DIR/<run_id>/...
        PROCESSED_DIR/<run_id>/feature_master*.parquet
        MODELS_DIR/<run_id>/...

    - In S3 mode:
        RAW_BUCKET:        <run_id>/...
        PROCESSED_BUCKET:  <run_id>/feature_master*.parquet
        MODELS_BUCKET:     <run_id>/...
    """
    info_by_id: Dict[str, RunInfo] = {}

    def _get(run_id: str) -> RunInfo:
        if run_id not in info_by_id:
            info_by_id[run_id] = RunInfo(run_id=run_id)
        return info_by_id[run_id]

    # ---------- Registered models (from global registry) ----------
    try:
        registry_entries = load_model_registry()
        registered_run_ids = {
            entry.get("run_id")
            for entry in registry_entries
            if entry.get("run_id") is not None
        }
    except Exception as ex:
        LOGGER.warning("Could not load model registry in list_runs: %s", ex)
        registered_run_ids = set()

    if _is_s3():
        # ---------- RAW ----------
        if SETTINGS.RAW_BUCKET:
            for key in list_bucket_objects(SETTINGS.RAW_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                run_id = parts[0]
                _get(run_id).has_raw = True

        # ---------- PROCESSED (feature master) ----------
        if SETTINGS.PROCESSED_BUCKET:
            for key in list_bucket_objects(SETTINGS.PROCESSED_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                run_id, filename = parts[0], parts[-1]
                info = _get(run_id)
                if filename.startswith("feature_master_cleaned_") and filename.endswith(".parquet"):
                    info.has_feature_master_cleaned = True
                elif filename.startswith("feature_master_") and filename.endswith(".parquet"):
                    # Avoid double-counting cleaned as raw
                    if "cleaned" not in filename:
                        info.has_feature_master = True

        # ---------- MODELS ----------
        if getattr(SETTINGS, "MODELS_BUCKET", None):
            for key in list_bucket_objects(SETTINGS.MODELS_BUCKET, prefix=""):
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                run_id = parts[0]
                info = _get(run_id)
                info.has_model = True
                if run_id in registered_run_ids:
                    info.has_registered_model = True

    else:
        raw_root = Path(SETTINGS.RAW_DIR)
        proc_root = Path(SETTINGS.PROCESSED_DIR)
        models_root = Path(getattr(SETTINGS, "MODELS_DIR", "")) if getattr(SETTINGS, "MODELS_DIR", None) else None

        # ---------- RAW ----------
        if raw_root.exists():
            for run_dir in raw_root.iterdir():
                if run_dir.is_dir() and any(run_dir.iterdir()):
                    _get(run_dir.name).has_raw = True

        # ---------- PROCESSED (feature master) ----------
        if proc_root.exists():
            for run_dir in proc_root.iterdir():
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                info = _get(run_id)

                fm_clean = list(run_dir.glob("feature_master_cleaned_*.parquet"))
                fm_raw = [
                    p for p in run_dir.glob("feature_master_*.parquet")
                    if "cleaned" not in p.name
                ]

                if fm_clean:
                    info.has_feature_master_cleaned = True
                if fm_raw:
                    info.has_feature_master = True

        # ---------- MODELS ----------
        if models_root and models_root.exists():
            for run_dir in models_root.iterdir():
                if run_dir.is_dir() and any(run_dir.iterdir()):
                    run_id = run_dir.name
                    info = _get(run_id)
                    info.has_model = True
                    if run_id in registered_run_ids:
                        info.has_registered_model = True

    infos = list(info_by_id.values())

    # Keep same ordering semantics as your old _list_runs
    infos.sort(key=lambda ri: ri.run_id, reverse=True)
    LOGGER.info(
        "Run infos (%s backend) → %s",
        "S3" if _is_s3() else "LOCAL",
        [(ri.run_id, ri.has_raw, ri.has_feature_master, ri.has_feature_master_cleaned, ri.has_model) for ri in infos],
    )
    return infos


# -------------------------------------------------------------------
# Sidebar menu builder
# -------------------------------------------------------------------
def get_nav() -> tuple[str, str]:
    with st.sidebar:
        # Logo + app header
        _section_header()

        st.markdown('<div class="sidebar-title">Pipeline Runs</div>', unsafe_allow_html=True)
        st.markdown('<div class="run-list-container">', unsafe_allow_html=True)

        runs = _list_runs()
        current = st.session_state.get("run_id", "")

        if runs:
            for run in runs:
                rid = run.run_id
                stage = _infer_stage(run)
                created = _format_created_from_run_id(rid)
                run_label = f"{rid} - {stage} - {created if created else ''}"
                row = st.container()
                with row:
                    # Note: the button is NOT a child of this div in Streamlit DOM,
                    # so we style via sidebar-wide selectors above.
                    row.markdown(
                        f'<div class="menu-row {"active" if rid == current else ""}">',
                        unsafe_allow_html=True
                    )
                    clicked = st.button(run_label, key=f"run_{rid}", use_container_width=True)
                    # st.markdown(f'<div class="menu-status">{meta}</div></div>', unsafe_allow_html=True)
                    if clicked:
                        st.session_state["run_id"] = rid
                        st.session_state.pop("staged_raw", None)
                        st.session_state.pop("_raw_preview", None)
                        _ensure_run_dirs(rid)
                        st.rerun()
        else:
            st.caption("No runs yet. Create a new pipeline to get started.")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="menu-separator"></div>', unsafe_allow_html=True)

        # New Button
        if st.button("New Pipeline", key="btn_new_run", use_container_width=True):
            new_id = _new_run_id()
            st.session_state["run_id"] = new_id
            st.session_state.pop("staged_raw", None)
            st.session_state.pop("_raw_preview", None)
            _ensure_run_dirs(new_id)
            st.rerun()

        if st.session_state.get("run_id"):
            return ("Pipeline Runs", "pipeline_hub")
        else:
            return ("Pipeline Runs", "home")
