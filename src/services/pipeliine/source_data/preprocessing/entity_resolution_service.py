from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.config_loader import load_config_from_file
from src.utils.data_io_utils import save_processed
from src.utils.ent_res_utils import id_entity_map, resolve_entities
from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger("entity_resolution")

# Output entity mapping file
ENTITY_MAPPING_FILENAME = "feature_dataset_entity_mapping_" + datetime.now().strftime("%Y%m%d_%H%M")


def create_entity_mapping(
        base_dir: str,
        base: pd.DataFrame,
        core: pd.DataFrame,
        cand: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    LOGGER.info("Starting entity mapping...")
    # tuning from config
    er_config = load_config_from_file(str(Path(__file__).parent.parent.parent.parent / "config/er_config.toml"))
    contingency_year = er_config["entity_resolution"]["contingency_year"]
    min_similarity_score = er_config["entity_resolution"]["min_similarity_score"]
    country_iso2_code_map = er_config["entity_resolution"]["iso3_to_iso2_map"]

    # First: ID-first deterministic mapping
    LOGGER.info("ID Mapping in-progress....")
    mapping = id_entity_map(base, core)
    LOGGER.info(f"ID Mapping completed successfully, mapped {len(mapping)} records")

    # Second: For rows still unmatched - year and territory mapping
    matched_ids = set(mapping["license_id"])
    base_left = base[~base["license_id"].isin(matched_ids)]

    if not base_left.empty:
        LOGGER.info(f"{len(base_left)} un-matched rows exists, Year and Territory mapping in-progress....")
        er_map = resolve_entities(
            base_left,
            cand,
            contingency_year,
            min_similarity_score,
            country_iso2_code_map
        )
        mapping = pd.concat([mapping, er_map], ignore_index=True)

    LOGGER.info("Mapping created successfully")
    hit = 100 * mapping["title_id"].notna().mean()
    LOGGER.info(
        f"Mapping stats: Total Count = {len(mapping)} -> ID Match Count = {(mapping['method'] == 'id_exact').sum()}, "
        f"Year and Territory Match Count = {(mapping['method'] != 'id_exact').sum()}, "
        f"Total Hit = {hit: .1f}")

    save_processed(mapping, base_dir, ENTITY_MAPPING_FILENAME)

    return mapping, ENTITY_MAPPING_FILENAME
