from __future__ import annotations

from typing import List

import streamlit as st

from src.services.security.auth_service import user_is_admin
from src.services.security.role_service import (
    get_all_roles,
    get_role_name_to_id_map,
    get_role_id_to_name_map,
)
from src.services.security.user_service import (
    get_all_users,
    upsert_user,
    delete_user,
)


def render_admin() -> None:
    """
    Security Administration screen.

    - View roles and their allowed pages.
    - View existing users and their roles.
    - Add / update / delete users.

    Only accessible to Admins (checked via role names).
    """
    role_ids = st.session_state.get("role_ids", [])

    if not user_is_admin(role_ids):
        st.error("You must be an Admin to access Security Administration.")
        return

    st.header("Security Administration")

    # --- Roles & permissions ---
    st.subheader("Roles & Permissions")
    roles = get_all_roles()
    for role in roles:
        st.markdown(f"**{role.name}** – {role.description}")
        st.caption("Allowed pages: " + ", ".join(role.allowed_page_keys))

    st.divider()

    # --- Existing users ---
    st.subheader("Users")

    role_id_to_name = get_role_id_to_name_map()
    users = get_all_users()

    if users:
        for user in users:
            role_labels = [role_id_to_name.get(rid, str(rid)) for rid in user.role_ids]
            st.write(f"**{user.username}** – roles: {', '.join(role_labels)}")
    else:
        st.caption("No users defined.")

    st.divider()

    # --- Add / update user ---
    st.subheader("Add / Update User")

    role_name_to_id = get_role_name_to_id_map()
    role_names: List[str] = list(role_name_to_id.keys())

    with st.form("user_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        selected_role_names: List[str] = st.multiselect("Roles", role_names)

        submitted = st.form_submit_button("Save user")

        if submitted:
            if not username or not password or not selected_role_names:
                st.error("Username, password and at least one role are required.")
            else:
                role_ids_to_assign = [role_name_to_id[name] for name in selected_role_names]
                upsert_user(username=username, password=password, role_ids=role_ids_to_assign)
                st.success("User saved.")
                st.experimental_rerun()

    st.divider()

    # --- Delete user ---
    st.subheader("Delete User")

    if users:
        username_to_delete = st.selectbox("Select user", [u.username for u in users])
        if st.button("Delete selected user"):
            delete_user(username_to_delete)
            st.success(f"User '{username_to_delete}' deleted.")
            st.experimental_rerun()
    else:
        st.caption("No users to delete.")
