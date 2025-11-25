from __future__ import annotations

import base64

import streamlit as st

from src.ui.common import logo_path, APP_NAME
from src.services.security.auth_service import (
    authenticate as authenticate_user,
    get_allowed_page_keys_for_roles,
)


def require_login() -> None:
    """
    UI-only login gate.

    - Delegates authentication and RBAC to services/security/*.
    - Stores username, role_ids, and allowed_page_keys in session_state.
    """
    if st.session_state.get("authenticated"):
        return

    # Simple vertical spacing to center-ish the login
    st.markdown("<div style='height: 12vh'></div>", unsafe_allow_html=True)
    _, center, _ = st.columns([1.5, 1, 1.5])

    with center:
        st.markdown(
            "<div style='padding: 1.25rem 1.5rem;'>",
            unsafe_allow_html=True,
        )

        # --- Logo + heading ---
        path = logo_path()
        app_heading = f"{APP_NAME} - Login"

        if path:
            svg_text = path.read_text(encoding="utf-8")
            b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
            st.markdown(
                f"""
                <div style="
                    background: #f3f4ff;
                    padding: 0.75rem 1rem;
                    border-radius: 0.75rem;
                    display: flex;
                    align-items: center;
                    gap: 0.6rem;
                    margin-bottom: 1.25rem;
                ">
                    <img src="data:image/svg+xml;base64,{b64}" width="36" />
                    <span style="font-weight: 600; font-size: 1rem;">
                        {app_heading}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background: #f3f4ff;
                    padding: 0.75rem 1rem;
                    border-radius: 0.75rem;
                    margin-bottom: 1.25rem;
                ">
                    <span style="font-weight: 600; font-size: 1rem;">
                        {app_heading}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- Login card ---
        with st.container(border=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

            if st.button("Login"):
                user = authenticate_user(username, password)
                if user is None:
                    st.error("Invalid username or password.")
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user["username"]
                    st.session_state["display_name"] = user.get("name") or user["username"]
                    st.session_state["role_ids"] = user["role_ids"]
                    st.session_state["allowed_page_keys"] = user["allowed_page_keys"]

                    st.success("Login successful.")
                    st.rerun()

    # Do not render any other pages until authenticated
    st.stop()
