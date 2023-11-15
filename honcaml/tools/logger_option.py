import yaml


def yaml_reader(config_path):
    """
    This function reads the command-line to extract the name of yaml
    configuration file


    Returns:
        config: yaml
    """
    config = None
    if config_path is not None:
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

    return config


def logger_opt(config):
    """
    This function sets up logging level options based on configuration file
    from the user.

    Args:
        config: yaml

    Returns:
        log: str
    """
    options = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    log = None
    if config is not None:
        if 'logging' in config:
            log = config['logging']['level']
            log = log.upper()
            if log not in options:
                log = None

    return log
