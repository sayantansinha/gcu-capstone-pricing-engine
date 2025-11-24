from __future__ import annotations

import base64
from typing import Tuple

import streamlit as st

from src.config.page_constants import PAGE_KEY_HOME, PAGE_KEY_PIPELINE, PAGE_KEY_PRC_PRED, PAGE_KEY_CMPR_BNDL
from src.ui.common import logo_path, APP_NAME

_MENU_LABEL_HOME = "Home"
_MENU_LABEL_PIPELINE = "Pipeline"
_MENU_LABEL_PRC_PRED = "Price Predictor"
_MENU_LABEL_CMPR_BNDL = "Compare & Bundling"


def _section_header() -> None:
    """
    Render logo + app title at the top of the sidebar.
    """
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


def _main_menu_button(label: str, key: str, active: bool) -> bool:
    """
    Render a single main menu button row.

    Returns:
        bool: True if this button was clicked this run.
    """
    row = st.container()
    css_class = "menu-main-row active" if active else "menu-main-row"
    with row:
        row.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
        clicked = st.button(label, key=key, use_container_width=True)
    return clicked


def get_nav() -> Tuple[str, str]:
    """
    Build the sidebar navigation.

    Returns:
        (active_main_label, page_key)

    Examples of page_key:
        - "home"
        - "pipeline_hub"
        - "price_predictor"
        - "compare_bundling"
        - "trend_analysis"
        - "what_if_simulator"
        - "reports"
    """
    with st.sidebar:
        _section_header()

        # Restore current navigation state from session
        active_main = st.session_state.get("nav_main", _MENU_LABEL_HOME)
        page_key = st.session_state.get("nav_page_key", PAGE_KEY_HOME)

        # -------------------------
        # Home
        # -------------------------
        if _main_menu_button(
                label=_MENU_LABEL_HOME,
                key="nav_main_home",
                active=(active_main == _MENU_LABEL_HOME),
        ):
            active_main = _MENU_LABEL_HOME
            page_key = PAGE_KEY_HOME
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        # -------------------------
        # Pipeline
        # -------------------------
        if _main_menu_button(
                label=_MENU_LABEL_PIPELINE,
                key="nav_main_pipeline",
                active=(active_main == _MENU_LABEL_PIPELINE),
        ):
            active_main = _MENU_LABEL_PIPELINE
            page_key = PAGE_KEY_PIPELINE
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        # -------------------------
        # Price Predictor
        # -------------------------
        if _main_menu_button(
                label=_MENU_LABEL_PRC_PRED,
                key="nav_main_price_predictor",
                active=(active_main == _MENU_LABEL_PRC_PRED),
        ):
            active_main = _MENU_LABEL_PRC_PRED
            page_key = PAGE_KEY_PRC_PRED
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        # -------------------------
        # Compare & Bundling
        # -------------------------
        if _main_menu_button(
                label=_MENU_LABEL_CMPR_BNDL,
                key="nav_main_compare_bundling",
                active=(active_main == _MENU_LABEL_CMPR_BNDL),
        ):
            active_main = _MENU_LABEL_CMPR_BNDL
            page_key = PAGE_KEY_CMPR_BNDL
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        # -------------------------
        # Trend Analysis
        # -------------------------
        if _main_menu_button(
                label="Trend Analysis",
                key="nav_main_trend_analysis",
                active=(active_main == "Trend Analysis"),
        ):
            active_main = "Trend Analysis"
            page_key = "trend_analysis"
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        # -------------------------
        # What-If Simulator
        # -------------------------
        if _main_menu_button(
                label="What-If Simulator",
                key="nav_main_what_if",
                active=(active_main == "What-If Simulator"),
        ):
            active_main = "What-If Simulator"
            page_key = "what_if_simulator"
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        # -------------------------
        # Reports
        # -------------------------
        if _main_menu_button(
                label="Reports",
                key="nav_main_reports",
                active=(active_main == "Reports"),
        ):
            active_main = "Reports"
            page_key = "reports"
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        return active_main, page_key
