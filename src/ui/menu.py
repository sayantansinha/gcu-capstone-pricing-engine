from __future__ import annotations

import base64
from typing import Tuple

import streamlit as st

from src.config.page_constants import (
    PAGE_KEY_HOME,
    PAGE_KEY_PIPELINE,
    PAGE_KEY_PRC_PRED,
    PAGE_KEY_CMPR_BNDL,
    PAGE_KEY_RPT,
    PAGE_KEY_SECURITY_ADMIN,
)
from src.services.security.role_service import LOGGER
from src.ui.common import logo_path, APP_NAME

_MENU_LABEL_HOME = "Home"
_MENU_LABEL_PIPELINE = "Pipeline"
_MENU_LABEL_PRC_PRED = "Price Predictor"
_MENU_LABEL_CMPR_BNDL = "Compare & Bundling"
_MENU_LABEL_RPT = "Reports"
_MENU_LABEL_ADMIN = "Admin"


def _section_header(display_name: str) -> None:
    """
    Render logo + app title + signed-in text + logout icon button.
    """
    # --- Logo + App Name ---
    path = logo_path()
    if path:
        svg_text = path.read_text(encoding="utf-8")
        b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        st.sidebar.markdown(
            f"""
            <div class="logo-container">
                <img src="data:image/svg+xml;base64,{b64}" />
                <span class="app-name-text">{APP_NAME}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(f"### {APP_NAME}")

    # --- small space between header and user row ---
    st.sidebar.markdown(
        "<div style='height:0.35rem;'></div>", unsafe_allow_html=True
    )

    # --- Signed in as + logout icon ---
    if display_name:
        user_col, logout_col = st.sidebar.columns([4, 1])

        with user_col:
            st.markdown(
                f"<div class='sidebar-user'>Signed in as <b>{display_name}</b></div>",
                unsafe_allow_html=True,
            )

        with logout_col:
            if st.button("⎋", key="logout_btn", help="Logout"):
                # Clear authentication-related session keys
                for key in [
                    "authenticated",
                    "username",
                    "display_name",
                    "role_ids",
                    "allowed_page_keys",
                    "nav_main",
                    "nav_page_key",
                ]:
                    st.session_state.pop(key, None)

                st.rerun()

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


def _can_show(page_key: str) -> bool:
    """
    Whether the current user should see a menu item for the given page key.

    Uses allowed_page_keys computed during login and stored in session.
    If not present (e.g. legacy or tests), we default to 'show all'.
    """
    allowed = st.session_state.get("allowed_page_keys")
    if not allowed:
        return True
    return page_key in allowed


def get_nav() -> Tuple[str, str]:
    """
    Build the sidebar navigation.

    Returns:
        (active_main_label, page_key)
    """
    display_name = st.session_state.get("display_name") or st.session_state.get("username")

    with st.sidebar:
        _section_header(display_name)

        # Restore current navigation state from session
        active_main = st.session_state.get("nav_main", _MENU_LABEL_HOME)
        page_key = st.session_state.get("nav_page_key", PAGE_KEY_HOME)

        # If current page is not allowed anymore (e.g. roles changed),
        # fall back to Home.
        allowed = st.session_state.get("allowed_page_keys")
        LOGGER.debug(f"allowed list: {allowed}")
        if allowed and page_key not in allowed:
            LOGGER.warning("Page key '%s' is not allowed for user '%s", page_key, display_name)
            active_main = _MENU_LABEL_HOME
            page_key = PAGE_KEY_HOME
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key

        # -------------------------
        # Home (always visible; also present in allowed_page_keys for all roles
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
        if _can_show(PAGE_KEY_PIPELINE) and _main_menu_button(
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
        if _can_show(PAGE_KEY_PRC_PRED) and _main_menu_button(
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
        if _can_show(PAGE_KEY_CMPR_BNDL) and _main_menu_button(
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
        # Reports
        # -------------------------
        if _can_show(PAGE_KEY_RPT) and _main_menu_button(
                label=_MENU_LABEL_RPT,
                key="nav_main_reports",
                active=(active_main == _MENU_LABEL_RPT),
        ):
            active_main = _MENU_LABEL_RPT
            page_key = PAGE_KEY_RPT
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        # -------------------------
        # Security Admin
        # -------------------------
        if _can_show(PAGE_KEY_SECURITY_ADMIN) and _main_menu_button(
                label=_MENU_LABEL_ADMIN,
                key="nav_main_admin",
                active=(active_main == _MENU_LABEL_ADMIN),
        ):
            active_main = _MENU_LABEL_ADMIN
            page_key = PAGE_KEY_SECURITY_ADMIN
            st.session_state["nav_main"] = active_main
            st.session_state["nav_page_key"] = page_key
            st.rerun()

        return active_main, page_key
