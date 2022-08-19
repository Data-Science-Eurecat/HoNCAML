from src import exceptions
from typing import Dict, Callable
import datetime
import uuid
import importlib


def import_library(module: str, params: Dict = None) -> Callable:
    """
    Given a module name and dict params, this function imports the module and
    creates a new callable with specific parameters.

    Args:
        module (str): module name.
        params (Dict): dict that contains the parameters for the specific
            module initialization.

    Returns:
        callable of imported module with parameters.
    """
    library = '.'.join(module.split('.')[:-1])
    imported_module = importlib.import_module(library)
    name = module.split('.')[-1]

    if params is None:
        params = dict()

    return getattr(imported_module, name)(**params)


def ensure_input_list(obj: object) -> list:
    """
    Ensure that input is a list; if not, return one.

    Args:
        obj: Any object.

    Returns:
        lst: Returned list.

    """
    if isinstance(obj, list):
        lst = obj
    elif not obj:
        lst = []
    else:
        lst = [obj]
    return lst


def generate_unique_id(estimator_name: str, adding_uuid: bool = False) -> str:
    """
    This function generates a unique string id based on current timestamp and
    uuid4.

    Args:
        estimator_name (str):
        adding_uuid (optional, bool): adding uuid4 in generated id.

    Returns:
        a unique id string.
    """
    unique_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if adding_uuid:
        unique_id = f'{unique_id}_{uuid.uuid4()}'

    return f'{estimator_name}.{unique_id}'


def validate_pipeline(pipeline_content: Dict) -> None:
    """
    Validate the pipeline steps based on the rules defined to prevent invalid
    executions.

    Args:
        pipeline_content (Dict): the settings defining the pipeline steps.
    """
    # TODO: loop the steps and check the rules defined by the settings.yaml file: params['pipeline_rules']
    # Raise an exception when the rule validation fail


def merge_settings(base_settings: Dict, user_settings: Dict,
                   acc_key: str = '') -> Dict:
    """
    Update the base settings with the user defined settings recursively.

    Args:
        base_settings (Dict): the library base settings.
        user_settings (Dict): the user modified settings.
        acc_key (str): the accumulated key path from the settings.

    Returns:
        base_settings (Dict): the base settings updated with the user ones.
    """
    for key in user_settings:
        acc_key = f'{acc_key}.{key}' if acc_key != '' else f'{key}'
        if key in base_settings:
            if isinstance(user_settings[key], dict):
                base_settings[key] = merge_settings(
                    base_settings[key], user_settings[key], acc_key)
            else:
                base_settings[key] = user_settings[key]
        else:
            raise exceptions.settings.SettingsDoesNotExist(acc_key)
    return base_settings
