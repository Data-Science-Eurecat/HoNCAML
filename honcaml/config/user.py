import os
import shutil

from honcaml.tools.startup import logger

PATH_HERE = os.path.abspath(os.path.dirname(__file__))


def export_config(
        user_level: str, filepath: str, pipeline_type: str) -> None:
    """
    Export a YAML configuration to a specified path.

    Args:
        user_level: Knowledge level of user.
        filepath: Path where to save the basic configuration file.
        pipeline_type: Type of pipeline to use.
    """
    logger.info('Generating configuration file')
    templates_dir = os.path.join(PATH_HERE, "templates")
    filename = '_'.join([user_level, pipeline_type]) + '.yaml'
    config_file = os.path.join(templates_dir, filename)
    shutil.copyfile(config_file, filepath)
    logger.info('Configuration file generated')
