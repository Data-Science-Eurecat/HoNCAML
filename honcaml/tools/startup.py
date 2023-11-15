import datetime
import logging
import sys

from honcaml.config.default_params import params
from honcaml.tools.logger_option import logger_opt
# Load settings
params['exec_name'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

log_params = params['logging']
logger = logging.getLogger(__name__)
hdlr_out = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(log_params['formatter']['format'],
                              log_params['formatter']['time_format'])
hdlr_out.setFormatter(formatter)
logger.addHandler(hdlr_out)
logger.setLevel(getattr(logging, log_params['level']))
logger.propagate = False


def setup_logging_config(config: str) -> None:
    """
    Read the config file and define the logger level if is possible

    Args:
        config: dict
    """
    log_config = logger_opt(config)
    if log_config is not None:
        logger.setLevel(getattr(logging, log_config))


def setup_file_logging(filepath: str) -> None:
    """
    Adds a file handler to the logger given an specific filepath.

    Args:
        filepath: the file where the logs will be stored.
    """
    hdlr_file = logging.FileHandler(filepath)
    hdlr_file.setFormatter(formatter)
    logger.addHandler(hdlr_file)
