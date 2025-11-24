import toml

from src.utils.log_utils import get_logger

LOGGER = get_logger("config_loader")


def load_config_from_file(config_file: str) -> dict:
    """
    Load TOML configuration file
    :param config_file: toml config file path relevant to the root directory of the project
    :return: dictionary representation of the config file
    """
    res = toml.load(config_file)
    LOGGER.info(f"Loaded config file {config_file} :: {res}")
    return res
