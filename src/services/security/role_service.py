from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

from src.utils.log_utils import get_logger
from src.utils.security_utils import load_rbac_config, save_rbac_config

LOGGER = get_logger("role_service")


@dataclass
class Role:
    id: int
    name: str
    description: str
    allowed_page_keys: List[str]


@lru_cache(maxsize=1)
def _load_roles() -> List[Role]:
    raw = load_rbac_config()
    roles: List[Role] = []
    for item in raw:
        roles.append(
            Role(
                id=int(item["id"]),
                name=item["name"],
                description=item.get("description", ""),
                allowed_page_keys=list(item.get("allowed_page_keys", [])),
            )
        )
    return roles


def get_all_roles() -> List[Role]:
    return list(_load_roles())


def get_role_by_id(role_id: int) -> Optional[Role]:
    for role in _load_roles():
        if role.id == role_id:
            return role
    return None


def get_roles_for_ids(role_ids: List[int]) -> List[Role]:
    wanted = set(int(rid) for rid in role_ids)
    return [r for r in _load_roles() if r.id in wanted]


def get_allowed_page_keys_for_role_ids(role_ids: List[int]) -> List[str]:
    allowed: set[str] = set()
    for role in get_roles_for_ids(role_ids):
        allowed.update(role.allowed_page_keys)
    LOGGER.debug("Allowed page keys: %s", allowed)
    return sorted(allowed)


def get_role_name_to_id_map() -> Dict[str, int]:
    return {r.name: r.id for r in _load_roles()}


def get_role_id_to_name_map() -> Dict[int, str]:
    return {r.id: r.name for r in _load_roles()}


# (Optional) if you ever want to edit roles through UI:
def save_roles(roles: List[Role]) -> None:
    raw = [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "allowed_page_keys": r.allowed_page_keys,
        }
        for r in roles
    ]
    save_rbac_config(raw)
    _load_roles.cache_clear()
