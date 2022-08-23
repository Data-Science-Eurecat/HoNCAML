import datetime
import importlib
import uuid
from typing import Dict, Callable

from src.exceptions import settings as settings_exception


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


def generate_unique_id(
        estimator_name: str = None, adding_uuid: bool = False) -> str:
    """
    This function generates a unique string id based on current timestamp and
    uuid4.

    Args:
        estimator_name (str): name of estimator that pipeline contains
        adding_uuid (optional, bool): adding uuid4 in generated id.

    Returns:
        a unique id string.
    """
    unique_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if adding_uuid:
        unique_id = f'{unique_id}_{uuid.uuid4()}'

    if estimator_name:
        unique_id = f'{estimator_name}.{unique_id}'

    return unique_id


def validate_pipeline(pipeline_content: Dict) -> None:
    """
    Validate the pipeline steps based on the rules defined to prevent invalid
    executions.

    Args:
        pipeline_content (Dict): the settings defining the pipeline steps.
    """
    # TODO: loop the steps and check the rules defined by the settings.yaml file: params['pipeline_rules']
    # Raise an exception when the rule validation fail
    pass


def merge_settings(
        base_settings: Dict, user_settings: Dict, acc_key: str = '') -> Dict:
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
            raise settings_exception.SettingParameterDoesNotExist(acc_key)
    return base_settings


def update_dict_from_default_dict(
        source_dict: Dict, overrides_dict: Dict) -> Dict:
    """
    Given two dictionaries, this function combine both dictionaries values
    and 'overrides_dict' values prevails. In addition, it is a recursive
    function when the value of dict is another dict.

    Args:
        source_dict (Dict): dictionary to modify
        overrides_dict (Dict): dictionary with override values

    Returns:
        a dict with values of both dicts.
    """
    for key, value in overrides_dict.items():
        if isinstance(value, dict) and value:
            returned = update_dict_from_default_dict(
                source_dict.get(key, {}), value)
            source_dict[key] = returned
        else:
            source_dict[key] = overrides_dict[key]

    return source_dict
