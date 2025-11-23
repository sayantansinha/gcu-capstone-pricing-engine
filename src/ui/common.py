import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.log_utils import get_logger

LOGGER = get_logger("ui_common")

# App identity (logo + name)
APP_NAME = "Predictive Pricing Engine"
_LOGO_PATHS = [
    "src/ui/assets/logo.svg",
    "ui/assets/logo.svg",
    "assets/logo.svg",
    "logo.svg"
]


def logo_path() -> Path | None:
    for p in _LOGO_PATHS:
        if os.path.exists(p):
            return Path(p)
    return None


def inject_css_from_file(css_path: str, rerun_on_first_load: bool = True):
    """
    Inject CSS globally and guarantee it applies even on the first render.

    Parameters
    ----------
    css_path : str
        Path to the CSS file.
    rerun_on_first_load : bool, default True
        Whether to perform a one-time rerun after first CSS injection
        (ensures proper styling on the very first load).
    """
    state_key_prefix: str = "_css_injected"
    injected_key = f"{state_key_prefix}_injected"
    rerun_key = f"{state_key_prefix}_rerun_done"
    mtime_key = f"{state_key_prefix}_mtime"

    # If already injected, optionally check for file change
    # if st.session_state.get(injected_key, False):
    #     if autorefresh:
    #         p = Path(css_path).expanduser()
    #         if p.exists():
    #             mtime = p.stat().st_mtime
    #             if st.session_state.get(mtime_key) != mtime:
    #                 st.session_state[mtime_key] = mtime
    #                 st.session_state[rerun_key] = True
    #                 st.rerun()
    #     return

    # Load and inject CSS
    p = Path(css_path).expanduser()
    if not p.exists():
        st.warning(f"CSS file not found: {p}")
        st.session_state[injected_key] = True
        return

    css_text = p.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

    # Track mtime for dev autorefresh
    st.session_state[mtime_key] = os.path.getmtime(p)

    # Mark injected and optionally rerun once
    st.session_state[injected_key] = True
    if rerun_on_first_load and not st.session_state.get(rerun_key, False):
        st.session_state[rerun_key] = True
        st.rerun()


@contextlib.contextmanager
def noop_container():
    """A no-op context manager that behaves like a layout container."""
    yield


def section_panel(title: str, expanded: bool = False):
    """
    Standard section wrapper. By default, renders an expander.
    If st.session_state['_suppress_section_panel'] is True, render a simple container instead.
    This allows parent pages (like the Pipeline Hub) to avoid nested expanders.
    """
    if st.session_state.get("_suppress_section_panel", False):
        # render without an expander to avoid nesting
        return st.container()
    # default behavior (your original pattern)
    return st.expander(title, expanded=expanded)


def begin_tab_scroll():
    """Start a fixed-height, scrollable area inside a tab."""
    st.markdown("<div class='tab-scroll'>", unsafe_allow_html=True)


def end_tab_scroll():
    """Close the scrollable area."""
    st.markdown("</div>", unsafe_allow_html=True)


def get_run_id_from_session_state() -> str:
    """Get Rub ID from the session"""
    return st.session_state["run_id"]


def load_active_feature_master_from_session():
    p = st.session_state.get("last_feature_master_path")

    if not p and not os.path.exists(p):
        LOGGER.warning("No active feature master found in session")
        return None

    return pd.read_parquet(p), os.path.basename(p)


def load_active_cleaned_feature_master_from_session():
    p = st.session_state.get("last_feature_master_path")

    if not p or not os.path.exists(p):
        LOGGER.warning("No active feature master found in session")
        return None, None

    return pd.read_parquet(p), os.path.basename(p)


def extract_last_trained_models(formatted: bool = False):
    if st.session_state.get("last_model"):
        last_trained_models = st.session_state.get("last_model")["trained_models"]
        if last_trained_models and formatted:
            return ', '.join(last_trained_models)
        else:
            return last_trained_models
    else:
        return None


def show_last_training_badge():
    last_trained_models = extract_last_trained_models(True)
    if last_trained_models:
        st.success(f"Last trained models: **{last_trained_models}**")


def store_last_model_info_in_session(
        base: dict,
        stacked: dict,
        comb_avg: dict,
        comb_wgt: dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pred_source: str,
        params_map: dict,
        trained_models: list[str],
        model_name: str
):
    st.session_state["last_model"] = {
        "base": base,
        "ensemble_stacked": stacked,
        "ensemble_avg": comb_avg,
        "ensemble_wgt": comb_wgt,
        "y_true": y_true,
        "y_pred": y_pred,
        "pred_source": pred_source,
        "params_map": params_map,
        "trained_models": trained_models,
        "model_name": model_name
    }


def store_last_run_model_dir_in_session(run_dir: str = None):
    """Store the last run model directory location for display"""
    st.session_state["last_model_run_dir"] = run_dir
