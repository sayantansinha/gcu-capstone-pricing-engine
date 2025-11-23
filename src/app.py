from __future__ import annotations

import sys

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from src.config.env_loader import SETTINGS
from src.ui.common import inject_css_from_file
from src.ui.auth import require_login
from src.ui.menu.menu import get_nav
from src.ui.pipeline.pipeline_hub import render as render_pipeline_hub
from src.utils.log_utils import handle_streamlit_exception, get_logger

sys.excepthook = handle_streamlit_exception
LOGGER = get_logger("app")

st.set_page_config(
    page_title="Predictive Pricing Engine",
    page_icon="ui/assets/logo.svg",
    layout="wide"
)

# dev mode: ensure css changes triggers an automatic reload
ctx = get_script_run_ctx()
if ctx and ctx.session_id:
    inject_css_from_file("src/ui/styles/main_app.css")

# --- session defaults ---
for k, v in {
    "df": None,
    "run_id": None,
    "raw_path": None,
    "steps": [],
    "ingested": False,
    "_show_viz_panel": False,
}.items():
    st.session_state.setdefault(k, v)


# --- inline blank-slate renderer ---
def _render_home():
    st.title("Welcome")
    st.subheader("Start a pipeline")
    st.write(
        "Use **New Pipeline** at the bottom-right of the sidebar to create a fresh run, "
        "or click an existing run under **Pipeline Runs** to continue."
    )
    st.caption(
        "After creating/selecting a run, the main area will guide you through staging, "
        "feature building, EDA, cleaning, and modeling."
    )


# --- routes ---
ROUTES = {
    "home": _render_home,
    "pipeline_hub": render_pipeline_hub,
}


def _dispatch(page_key: str):
    ROUTES.get(page_key, _render_home)()


def main():
    # --- auth gate ---
    if SETTINGS.IO_BACKEND != "LOCAL":
        require_login()

    # --- sidebar + route ---
    _, page_key = get_nav()
    _dispatch(page_key)


if __name__ == "__main__":
    main()
