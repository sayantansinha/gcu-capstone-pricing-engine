from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from src.utils.security_utils import (
    load_users_config,
    save_users_config,
)


@dataclass
class User:
    username: str
    password: str
    role_ids: List[int]
    name: str
    last_login_utc: Optional[str] = None
    recent_activities: List[str] = field(default_factory=list)


_users_cache: List[User] | None = None


def _load_users_from_storage() -> List[User]:
    raw = load_users_config()
    users: List[User] = []
    for item in raw:
        users.append(
            User(
                username=item["username"],
                password=item["password"],
                role_ids=[int(r) for r in item.get("roles", [])],
                name=item.get("name", item["username"]),
                last_login_utc=item.get("last_login_utc"),
                recent_activities=list(item.get("recent_activities", [])),
            )
        )
    return users


def _ensure_cache() -> List[User]:
    global _users_cache
    if _users_cache is None:
        _users_cache = _load_users_from_storage()
    return _users_cache


def get_all_users() -> List[User]:
    return list(_ensure_cache())


def find_user_by_username(username: str) -> Optional[User]:
    for user in _ensure_cache():
        if user.username == username:
            return user
    return None


def _flush_to_storage(users: List[User]) -> None:
    raw = [
        {
            "username": u.username,
            "password": u.password,
            "name": u.name,
            "roles": [int(r) for r in u.role_ids],
            "last_login_utc": u.last_login_utc,
            "recent_activities": list(u.recent_activities),
        }
        for u in users
    ]
    save_users_config(raw)


def upsert_user(username: str, password: str, role_ids: List[int], name: Optional[str] = None) -> User:
    """
    Create or update a user.

    - If username exists, overwrite password, roles, and name.
    - Otherwise, append a new user with empty metadata.
    """
    global _users_cache
    users = _ensure_cache()
    display_name = name or username

    for idx, user in enumerate(users):
        if user.username == username:
            users[idx] = User(
                username=username,
                password=password,
                role_ids=[int(r) for r in role_ids],
                name=display_name,
                last_login_utc=user.last_login_utc,
                recent_activities=list(user.recent_activities),
            )
            _flush_to_storage(users)
            _users_cache = users
            return users[idx]

    new_user = User(
        username=username,
        password=password,
        role_ids=[int(r) for r in role_ids],
        name=display_name,
    )
    users.append(new_user)
    _flush_to_storage(users)
    _users_cache = users
    return new_user


def delete_user(username: str) -> None:
    global _users_cache
    users = _ensure_cache()
    users = [u for u in users if u.username != username]
    _flush_to_storage(users)
    _users_cache = users


def update_last_login(username: str) -> None:
    """
    Set last_login_utc for the user to current UTC time.
    """
    global _users_cache
    users = _ensure_cache()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for u in users:
        if u.username == username:
            u.last_login_utc = now
            break
    _flush_to_storage(users)
    _users_cache = users


def log_user_activity(username: str, activity: str, max_items: int = 20) -> None:
    """
    Append a new activity entry to recent_activities, keeping only last max_items.

    Activity is stored as '<ISO_UTC_TIMESTAMP> - <activity>'.
    """
    global _users_cache
    users = _ensure_cache()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = f"{ts} - {activity}"

    for u in users:
        if u.username == username:
            u.recent_activities.append(entry)
            if len(u.recent_activities) > max_items:
                u.recent_activities = u.recent_activities[-max_items:]
            break

    _flush_to_storage(users)
    _users_cache = users
