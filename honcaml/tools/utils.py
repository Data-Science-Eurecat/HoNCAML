import datetime
import importlib
import uuid
from cerberus import Validator
from functools import reduce
from typing import Dict, Callable


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
    name = module.split('.')[-1]
    imported_module = importlib.import_module(library)

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
        estimator_module: str = None, estimator_type: str = None,
        adding_uuid: bool = False) -> str:
    """
    This function generates a unique string id based on current timestamp and
    uuid4.

    Args:
        estimator_module (str): name of estimator that pipeline contains
        estimator_type (str): type of estimator that pipeline contains
        adding_uuid (optional, bool): adding uuid4 in generated id.

    Returns:
        a unique id string.
    """
    unique_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if adding_uuid:
        unique_id = f'{unique_id}_{uuid.uuid4()}'

    if estimator_type:
        unique_id = f'{estimator_type}.{unique_id}'

    if estimator_module:
        unique_id = f'{estimator_module}.{unique_id}'

    return unique_id


def update_dict_from_default_dict(
        default_dict: Dict, source_dict: Dict) -> Dict:
    """
    Given two dictionaries combine both dictionary values having source_dict
    values prevailing over default ones. In addition, it throws a recursion
    over the dictionary values.

    Args:
        default_dict (Dict): dictionary with the default values.
        source_dict (Dict): dictionary to be updated.

    Returns:
        source_dict (Dict): a dict with merged values.
    """
    if source_dict is None:
        source_dict = {}
    for key, value in default_dict.items():
        if key not in source_dict:
            if not isinstance(value, dict) and value is not None:
                source_dict[key] = value
        else:
            if isinstance(value, dict):
                source_dict[key] = update_dict_from_default_dict(
                    value, source_dict[key])
    return source_dict


def build_validator(rules: Dict) -> Validator:
    """
    Builds a cerberus.Validator object by creating a rule schema based on the
    rules given by parameter as a dictionary.

    Args:
        rules (Dict): the rules to validate the input with.

    Returns:
        validator (Validator): the validator build with the schema defined by
            the input rules.
    """
    schema = {}
    for key, value in rules.items():
        if isinstance(value, dict):
            schema[key] = build_validator_schema(rules[key])
        else:
            schema[key] = reduce(lambda a, b: {**a, **b}, value)

    return Validator(schema, allow_unknown=True)


def build_validator_schema(rules: Dict) -> Dict:
    """
    Build the schema for the validator given the rules as a dictionary.

    Args:
        rules (Dict): the rules to validate the input with.

    Returns:
        schema (Dict): the schema built with the rules given.
    """
    schema = {'type': 'dict'}
    for key, value in rules.items():
        if isinstance(value, dict):
            schema['keysrules'] = {'allowed': list(rules.keys())}
            schema['valuesrules'] = build_validator_schema(rules[key])
        else:
            if value is not None:
                schema['schema'] = schema.get('schema', {})
                schema['schema'][key] = reduce(lambda a, b: {**a, **b}, value)

    return schema


class FileExtension:
    """
    This class contains the available files formats to read data.
    """
    csv = '.csv'
    excel = ['.xlsx', '.xls']
    # Adding more file extensions here
