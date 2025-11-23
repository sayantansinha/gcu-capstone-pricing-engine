from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

from src.config.env_loader import SETTINGS
from src.config.config_loader import load_config_from_file
from src.utils.imdb_utils import load_imdb_core, load_imdb_akas, imdb_candidate_universe
from src.utils.title_utils import normalize_title
from src.utils.data_io_utils import save_processed, load_raw
from src.utils.log_utils import get_logger
from src.utils.feature_utils import preferred_column_order, validate_columns_exist_in_dataframe
from src.services.source_data.preprocessing.entity_resolution_service import create_entity_mapping

LOGGER = get_logger("feature_builder")
FEATURE_MASTER_FILENAME = "feature_master_" + datetime.now().strftime("%Y%m%d_%H%M")


def _load_raw_from_identifier(identifier: str, base_dir: str | None = None) -> pd.DataFrame:
    """
    Wrapper around load_raw() that understands:
      - LOCAL: full filesystem path
      - S3: s3://bucket/prefix/file.parquet  OR  'run_id/file.parquet'  OR just 'file' + base_dir
    """
    # S3 backend
    if SETTINGS.IO_BACKEND == "S3":
        # Full s3:// URI -> parse bucket/key and derive base_dir + name
        if identifier.startswith("s3://"):
            parsed = urlparse(identifier)
            key = parsed.path.lstrip("/")  # e.g. "20251117_123456/run_base.parquet"
            if "/" in key:
                derived_base_dir, file_part = key.split("/", 1)
                base_dir_eff = derived_base_dir
                name = Path(file_part).stem
            else:
                base_dir_eff = base_dir
                name = Path(key).stem
            return load_raw(name, base_dir=base_dir_eff)

        # Key-style identifier like "20251117_123456/run_base.parquet"
        if "/" in identifier and identifier.endswith(".parquet"):
            # For keys, we let load_raw treat it as the key directly
            return load_raw(identifier)

        # Bare filename (no prefix) -> need base_dir
        name = Path(identifier).stem
        return load_raw(name, base_dir=base_dir)

    # LOCAL backend: identifier is already a full path on disk
    return load_raw(identifier)


def _load_df_from_cache(raw_path: str) -> pd.DataFrame:
    LOGGER.info(f"Loading raw data from {raw_path}")
    cache_key = f"_raw_preview::{raw_path}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Use unified loader (handles LOCAL paths and S3 URIs/keys)
    df = _load_raw_from_identifier(raw_path)
    st.session_state[cache_key] = df
    return df


def label_staged_raw_files() -> Tuple[List, Dict[str, pd.DataFrame]]:
    staged_labels = list(st.session_state["staged_raw"].keys())
    LOGGER.info(f"Staged raw file labels: {staged_labels}")
    st.session_state["staged_files_count"] = len(staged_labels)
    label_to_df: Dict[str, pd.DataFrame] = {}
    for lbl in staged_labels:
        raw_path = st.session_state["staged_raw"][lbl]
        try:
            df = _load_df_from_cache(raw_path)
            label_to_df[lbl] = df
        except Exception as e:
            st.error(f"Failed reading RAW file for {lbl}: {e}")

    return staged_labels, label_to_df


def _rank_entity_map(emap: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple mapping rows exist per license_id, prefer stronger methods and higher scores.
    """
    # tuning from config
    er_config = load_config_from_file(str(Path(__file__).parent.parent.parent.parent / "config/er_config.toml"))
    method_rank = er_config["entity_resolution"]["method_rank"]
    emap = emap.assign(
        _mrank=emap["method"]
        .replace("+", "_")  # method has "+", replace with "_"
        .map(method_rank)
        .fillna(1)
        .astype(int)
    )
    emap = (
        emap.sort_values(["license_id", "_mrank", "match_score"], ascending=[True, False, False])
        .drop_duplicates("license_id", keep="first")
        .drop(columns=["_mrank"])
    )
    return emap


def build_features(
        base_dir: str,
        base_filename: str,
        imdb_basics_filename: str,
        imdb_ratings_filename: str,
        imdb_akas_filename: str
):
    # Load base (synthetic)
    # - LOCAL: base_filename is a full path (unchanged behavior)
    # - S3: base_filename may be a full s3:// URI, a key like "run_id/file.parquet",
    #       or just a bare name; base_dir is the run_id.
    base = _load_raw_from_identifier(base_filename, base_dir=base_dir)
    validate_columns_exist_in_dataframe(base, ["license_id", "title", "release_year"], "Synthetic base")
    LOGGER.info("Synthetic base file read and validated")

    # Normalized title
    base["title_norm"] = normalize_title(base["title"])
    LOGGER.info("Synthetic base file: title normalized")

    # IMDb candidates
    core = load_imdb_core(imdb_basics_filename, imdb_ratings_filename)
    validate_columns_exist_in_dataframe(
        core,
        required=["title_id", "primaryTitle", "releaseYear", "primary_title_norm"],
        name="IMDB core"
    )

    akas = load_imdb_akas(imdb_akas_filename)
    validate_columns_exist_in_dataframe(
        akas,
        required=["title_id", "aka_title_norm", "region", "language"],
        name="IMDB akas"
    )

    cand = imdb_candidate_universe(core, akas)
    validate_columns_exist_in_dataframe(
        cand,
        required=["title_id", "title_norm", "releaseYear", "numVotes"],
        name="IMDB Candidate Universe"
    )
    LOGGER.info("IMDB data populated and validated successfully")

    # Load mapping
    emap, emap_filename = create_entity_mapping(base_dir, base, core, cand)
    LOGGER.info(f"Entity Mapping file name {emap_filename}")
    validate_columns_exist_in_dataframe(emap, ["license_id", "title_id", "method", "match_score"], "Entity mapping")

    # rank entity map
    emap = _rank_entity_map(emap)
    LOGGER.info("Entity mapping file read and ranked")

    try:
        # Handle IMDb core
        imdb = core[
            [
                "title_id",
                "primaryTitle",
                "primary_title_norm",
                "releaseYear",
                "titleType",
                "averageRating",
                "numVotes",
            ]
        ].drop_duplicates("title_id")

        ## Joins
        # base and mapping
        final_df = base.merge(
            emap[["license_id", "method", "match_score"]],
            on="license_id",
            how="left",
        )

        # add IMDB
        LOGGER.debug(f"final_df columns = {final_df.columns}")
        LOGGER.debug(f"imdb columns = {imdb.columns}")
        final_df = final_df.merge(imdb, on="title_id", how="left")

        # ---- features
        # Prefer IMDb title when available; otherwise retain synthetic display name
        final_df["title_final"] = final_df["primaryTitle"].fillna(final_df["title"])

        # IMDb transforms
        final_df["log1p_numVotes"] = np.log1p(final_df["numVotes"].fillna(0))

        # Coverage flags
        final_df["has_entity"] = final_df["title_id"].notna().astype(int)

        # order and write
        final_df = final_df[preferred_column_order(final_df)]

        LOGGER.info(
            f"Rows={len(final_df)} "
            f"| ER hit={(100 * final_df['has_entity'].mean()): .1f} "
            f"| writing → {FEATURE_MASTER_FILENAME}")
    except Exception as ex:
        LOGGER.exception("Error while creating feature master file")
        raise ValueError("Error while creating feature master file") from ex

    return save_processed(final_df, base_dir, FEATURE_MASTER_FILENAME)
