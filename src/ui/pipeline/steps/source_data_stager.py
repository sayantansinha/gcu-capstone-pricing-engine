from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from src.services.pipeline.source_data.feature.feature_builder_service import label_staged_raw_files
from src.ui.common import get_run_id_from_session_state
from src.utils.data_io_utils import save_raw, list_raw_files, save_from_url
from src.utils.log_utils import streamlit_safe, get_logger

LOGGER = get_logger("ui_source_data_stager")

REQUIRED_KEYS = ["title_id"]


def _ensure_staging() -> None:
    """Keep a session-local 'staged' map of RAW paths the user intends to use for the FM build."""
    st.session_state.setdefault("staged_raw", {})  # label -> full raw_path (local path or S3 URI)


def _quick_checks(df: pd.DataFrame) -> dict:
    """Return basic profiling info for a staged DataFrame, safely handling missing keys."""
    nan_pct = df.isna().mean().sort_values(ascending=False).head(10)
    dtypes = df.dtypes.astype(str)
    existing_keys = [k for k in REQUIRED_KEYS if k in df.columns]
    if existing_keys:
        dup_keys = int(df.duplicated(subset=existing_keys).sum())
        has_keys = True
    else:
        dup_keys = 0
        has_keys = False
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "nan_pct": nan_pct,
        "dtypes": dtypes,
        "has_keys": has_keys,
        "dup_keys": dup_keys,
    }


def _add_staged(label: str, raw_path: str) -> None:
    LOGGER.info(f"Adding session state [staged_raw] :: label [{label}], raw [{raw_path}]")
    st.session_state["staged_raw"][label] = raw_path


@streamlit_safe
def render():
    run_id = get_run_id_from_session_state()
    st.subheader("Stage Data")
    _ensure_staging()

    # AUTO-STAGE existing RAW files on resume using data_io_utils
    if not st.session_state["staged_raw"]:
        try:
            files = list_raw_files(run_id)  # backend-agnostic: local or S3
            staged_now = 0
            if files:
                for raw_path in files:
                    # raw_path may be a local path or an S3 URI; use basename for the label
                    raw_str = str(raw_path)
                    LOGGER.info(f"Processing raw file [{raw_str}]")
                    label = os.path.basename(raw_str)
                    if label not in st.session_state["staged_raw"]:
                        _add_staged(label, raw_str)
                        staged_now += 1
            if staged_now:
                LOGGER.info(f"Staged raw file(s): {staged_now}")
                st.info(f"Auto-staged {staged_now} RAW file(s) for run {run_id}.")
        except Exception as e:
            LOGGER.exception("Auto-stage failed")
            st.warning(f"Auto-stage skipped: {e}")

    # Add sources into RAW (upload / url) - staged
    mode = st.radio("Add Raw Data Sources", ["Upload Files", "Load from URL"], horizontal=True)

    if mode == "Upload Files":
        files = st.file_uploader(
            "Upload CSV/Parquet (multi-select)",
            type=["csv", "parquet", "pq"],
            accept_multiple_files=True,
        )
        if files:
            added = 0
            for f in files:
                try:
                    df = (
                        pd.read_parquet(io.BytesIO(f.read()))
                        if f.name.endswith((".parquet", ".pq"))
                        else pd.read_csv(f)
                    )
                    raw_path = save_raw(
                        df,
                        run_id,
                        os.path.splitext(f.name)[0],
                    )
                    _add_staged(f.name, raw_path)
                    added += 1
                except Exception as e:
                    st.error(f"Failed to save {f.name} to RAW: {e}")
            if added:
                st.success(f"Saved {added} file(s) to RAW and staged them.")
    else:
        url = st.text_input("Enter CSV/Parquet URL")
        if st.button("Load Data"):
            if not url:
                st.warning("Please enter a URL.")
            else:
                try:
                    raw_path = save_from_url(url, run_id)  # backend-agnostic
                    label = os.path.basename(url)
                    _add_staged(label, raw_path)
                    st.success(f"Saved and staged: {label}")
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")

    if not st.session_state["staged_raw"]:
        LOGGER.info("No staged raw files")
        st.session_state["staged_files_count"] = 0
        st.info("No staged sources yet. Upload/add URLs or pick from RAW.")
    else:
        # Data Insights for each staged RAW file
        st.subheader("Data Insights (staged files)")
        _, label_to_df = label_staged_raw_files()
        for lbl, df in label_to_df.items():
            checks = _quick_checks(df)
            LOGGER.debug(f"Label [{lbl}] :: check result :: [{checks}]")
            with st.container(border=True):
                st.caption(f"{lbl} — {checks['rows']} rows, {checks['cols']} cols")
                tab_nan, tab_dtypes, tab_preview = st.tabs(["Percentage NaNs (10)", "Dtypes", "Preview"])
                with tab_nan:
                    st.dataframe(checks["nan_pct"], use_container_width=True)
                with tab_dtypes:
                    st.dataframe(checks["dtypes"].to_frame("dtype"), use_container_width=True)
                with tab_preview:
                    st.dataframe(df.head(10), use_container_width=True)
