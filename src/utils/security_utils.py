from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from src.config.env_loader import SETTINGS
from src.utils.log_utils import get_logger
from src.utils.data_io_utils import is_s3
from src.utils.s3_utils import ensure_bucket, load_bucket_object, write_bucket_object

LOGGER = get_logger("security_io_utils")

RBAC_FILE_NAME = "rbac.json"
USERS_FILE_NAME = "users.json"


# -------------------------------------------------------------------
# Internal helpers – where to store/read security config
# -------------------------------------------------------------------
def _load_security_json(filename: str) -> Any:
    """
    Internal generic loader – not exported.

    - LOCAL → DATA_DIR/SECURITY_DIR/<filename>
    - S3    → SECURITY_BUCKET/<filename>
    """
    if is_s3():
        bucket = SETTINGS.SECURITY_BUCKET
        LOGGER.info("Loading security JSON (S3) ← s3://%s/%s", bucket, filename)
        # load_bucket_object already parses *.json into dict/list
        return load_bucket_object(bucket, filename)

    directory = Path(SETTINGS.SECURITY_DIR)
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Security configuration file '{filename}' not found at {path}. "
            "Place rbac.json and users.json under DATA_DIR/SECURITY_DIR."
        )

    LOGGER.info("Loading security JSON (LOCAL) ← %s", path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_security_json(filename: str, data: Any) -> None:
    """
    Internal generic saver – not exported.
    """
    if is_s3():
        bucket = SETTINGS.SECURITY_BUCKET
        payload = json.dumps(data, indent=2)
        LOGGER.info("Saving security JSON (S3) → s3://%s/%s", bucket, filename)
        write_bucket_object(
            bucket=bucket,
            key=filename,
            payload=payload,
            content_type="application/json",
        )
        return

    directory = Path(SETTINGS.SECURITY_DIR)
    path = directory / filename
    LOGGER.info("Saving security JSON (LOCAL) → %s", path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -------------------------------------------------------------------
# PUBLIC security-specific helpers
# -------------------------------------------------------------------
def load_rbac_config() -> List[dict]:
    """Return the raw RBAC config from rbac.json."""
    return list(_load_security_json(RBAC_FILE_NAME))


def save_rbac_config(config: List[dict]) -> None:
    """Persist the RBAC config back to rbac.json."""
    _save_security_json(RBAC_FILE_NAME, config)


def load_users_config() -> List[dict]:
    """Return the raw users config from users.json."""
    return list(_load_security_json(USERS_FILE_NAME))


def save_users_config(config: List[dict]) -> None:
    """Persist the users config back to users.json."""
    _save_security_json(USERS_FILE_NAME, config)
