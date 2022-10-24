import os
import shutil

from honcaml.tools.startup import logger

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(PATH_HERE, "templates")

BASIC_CONFIG_FILE = os.path.join(TEMPLATES_DIR, "basic.yaml")
ADVANCED_CONFIG_FILE = os.path.join(TEMPLATES_DIR, "advanced.yaml")


def export_config(config_type: str, filepath: str) -> None:
    """
    Export a YAML configuration to a specified path.

    Args:
        config_type: Type of configuration to export.
        filepath: Path where to save the basic configuration file.
    """
    logger.info(f'Generating {config_type} configuration file')
    file_to_export = globals()[f'{config_type.upper()}_CONFIG_FILE']
    shutil.copyfile(file_to_export, filepath)
    logger.info('Configuration file generated')
