from __future__ import annotations

from typing import Dict, List, Optional

from src.services.security.user_service import (
    User,
    find_user_by_username,
    update_last_login,
    log_user_activity,
)
from src.services.security.role_service import (
    get_allowed_page_keys_for_role_ids,
    get_roles_for_ids,
)
from src.utils.log_utils import get_logger

LOGGER = get_logger("auth_service")

def authenticate(username: str, password: str) -> Optional[Dict[str, object]]:
    """
    Authenticate a user against users.json.

    On success:
      - Update last_login_utc
      - Log a 'Logged in' activity
      - Return a dict with username, name, role_ids, allowed_page_keys
    """
    user: Optional[User] = find_user_by_username(username)
    if user is None:
        return None
    if user.password != password:
        return None

    # Update metadata
    update_last_login(user.username)
    log_user_activity(user.username, "Logged in")

    allowed_page_keys = get_allowed_page_keys_for_role_ids(user.role_ids)

    return {
        "username": user.username,
        "name": user.name,
        "role_ids": list(user.role_ids),
        "allowed_page_keys": allowed_page_keys,
    }


def get_allowed_page_keys_for_roles(role_ids: List[int]) -> List[str]:
    return get_allowed_page_keys_for_role_ids(role_ids)


def user_can_access_page(page_key: str, role_ids: List[int]) -> bool:
    if not role_ids:
        LOGGER.warning("No role_ids specified.")
        return False
    LOGGER.debug(f"Page: {page_key}, roles to check {role_ids}")
    allowed = set(get_allowed_page_keys_for_role_ids(role_ids))
    return page_key in allowed


def user_is_admin(role_ids: List[int]) -> bool:
    for role in get_roles_for_ids(role_ids):
        if role.name.lower() == "admin":
            return True
    return False
