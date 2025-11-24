from __future__ import annotations

import sys

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from src.config.env_loader import SETTINGS
from src.config.page_constants import PAGE_KEY_PIPELINE, PAGE_KEY_HOME, PAGE_KEY_PRC_PRED, PAGE_KEY_CMPR_BNDL
from src.ui.bundling.compare_and_bundling import render_compare_and_bundling
from src.ui.common import inject_css_from_file
from src.ui.auth import require_login
from src.ui.home.home import render_home
from src.ui.menu import get_nav
from src.ui.pipeline.pipeline_hub import render as render_pipeline_hub
from src.ui.price_predictor.price_predictor import render_price_predictor
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
    inject_css_from_file("src/ui/styles/app.css")
    inject_css_from_file("src/ui/styles/menu.css")
    inject_css_from_file("src/ui/styles/pipeline.css")
    inject_css_from_file("src/ui/styles/price_predictor.css")
    inject_css_from_file("src/ui/styles/home.css")

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

# --- routes ---
ROUTES = {
    PAGE_KEY_HOME: render_home,
    PAGE_KEY_PIPELINE: render_pipeline_hub,
    PAGE_KEY_PRC_PRED: render_price_predictor,
    PAGE_KEY_CMPR_BNDL: render_compare_and_bundling
}


def _dispatch(page_key: str):
    ROUTES.get(page_key, render_home)()


def main():
    # --- auth gate ---
    if SETTINGS.IO_BACKEND != "LOCAL":
        require_login()

    # --- sidebar + route ---
    _, page_key = get_nav()
    _dispatch(page_key)


if __name__ == "__main__":
    main()
