import ntpath
import os
import posixpath
from dataclasses import dataclass, fields
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from src.utils.env_utils import getenv
from src.utils.log_utils import get_logger

LOGGER = get_logger("env_loader")


def load_env():
    # Load base '.env' if present
    base_env = find_dotenv(filename=".env", usecwd=True)
    if base_env:
        load_dotenv(base_env, override=False)

    # Highest-priority explicit file via ENV_FILE (absolute or relative)
    explicit_env = os.getenv("ENV_FILE")
    if explicit_env:
        load_dotenv(explicit_env, override=True)
    else:
        # profile via APP_ENV -> "{profile}.env"
        profile = os.getenv("APP_ENV")  # values: local | aws | test | prod
        if profile:
            prof_env = find_dotenv(filename=f"{profile}.env", usecwd=True)
            if prof_env:
                load_dotenv(prof_env, override=True)


load_env()


def _pjoin(*parts: str) -> str:
    cleaned = []
    for p in parts:
        if not p:
            continue

        # Detect absolute paths on ANY OS
        is_abs = (
                p.startswith("/")  # macOS / Linux absolute path
                or ntpath.isabs(p)  # Windows absolute path
                or posixpath.isabs(p)  # just in case Windows-style slash
        )

        # If absolute, strip to relative component
        if is_abs:
            p = Path(p).name  # keep only the last segment ("raw", "models", etc.)

        cleaned.append(p)

    return str(Path(*cleaned).resolve())


@dataclass
class Settings:
    # IO backend
    IO_BACKEND: str = getenv("IO_BACKEND", "LOCAL").upper()

    # Local dirs (relative to DATA_DIR)
    DATA_DIR: str = getenv("DATA_DIR", "./data")
    RAW_DIR: str = _pjoin(DATA_DIR, getenv("RAW_DIR", "raw"))
    PROCESSED_DIR: str = _pjoin(DATA_DIR, getenv("PROCESSED_DIR", "processed"))
    PROFILES_DIR: str = _pjoin(DATA_DIR, getenv("PROFILES_DIR", "profiles"))
    FIGURES_DIR: str = _pjoin(DATA_DIR, getenv("FIGURES_DIR", "figures"))
    MODELS_DIR: str = _pjoin(DATA_DIR, getenv("MODELS_DIR", "models"))
    REPORTS_DIR: str = _pjoin(DATA_DIR, getenv("REPORTS_DIR", "reports"))

    # S3 buckets + prefixes
    RAW_BUCKET: str = getenv("RAW_BUCKET")
    PROCESSED_BUCKET: str = getenv("PROCESSED_BUCKET")
    PROFILES_BUCKET: str = getenv("PROFILES_BUCKET")
    FIGURES_BUCKET: str = getenv("FIGURES_BUCKET")
    MODELS_BUCKET: str = getenv("MODELS_BUCKET")
    REPORTS_BUCKET: str = getenv("REPORTS_BUCKET")
    FEATURE_MASTER_PREFIX: str = getenv("FEATURE_MASTER_PREFIX", "feature_master_")

    # AWS config
    AWS_ENDPOINT_URL: str = getenv("AWS_ENDPOINT_URL")
    AWS_REGION: str = getenv("AWS_REGION")
    AWS_ACCESS_KEY_ID: str = getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN: str = getenv("AWS_SESSION_TOKEN")


SETTINGS = Settings()

# Ensure local dirs only when using LOCAL backend
if SETTINGS.IO_BACKEND == "LOCAL":
    for field in fields(Settings):
        name = field.name
        value = getattr(SETTINGS, name)
        # Create directory if it does not exist
        if name.endswith("_DIR"):
            os.makedirs(value, exist_ok=True)
            LOGGER.info(f"{name}: {value}")
else:
    for field in fields(Settings):
        LOGGER.info(f"{field.name}: {getattr(SETTINGS, field.name)}")
