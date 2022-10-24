import argparse
import datetime
import importlib
import re
import uuid
from cerberus import Validator
from functools import reduce
from typing import Dict, Callable


def import_library(module: str, params: Dict = None) -> Callable:
    """
    Imports the module specified and creates a new callable with specific
    parameters.

    Args:
        module: Module name.
        params: Parameters for the specific module initialization.

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
        estimator_module: str = None, adding_uuid: bool = False) -> str:
    """
    Generates a unique string id based on current timestamp and uuid4.

    Args:
        estimator_module: Name of estimator that pipeline contains.
        adding_uuid: Whether to add uuid4 in generated id.

    Returns:
        A unique id string.
    """
    unique_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if adding_uuid:
        unique_id = f'{unique_id}_{uuid.uuid4()}'

    if estimator_module:
        unique_id = f'{estimator_module}.{unique_id}'

    return unique_id


def update_dict_from_default_dict(
        default_dict: Dict, source_dict: Dict) -> Dict:
    """
    Combines two configurations prevailing the second over the default one.
    In addition, it throws a recursion over the dictionary values.

    Args:
        default_dict: Default configuration.
        source_dict: New configuration.

    Returns:
        A configuration with merged values.
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
    Builds a cerberus.Validator object by creating a rule schema.

    Args:
        rules: Rules to validate the input with.

    Returns:
        The validator build with the schema defined by the input rules.
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
    Builds the schema for the validator.

    Args:
        rules: Rules to validate the input with.

    Returns:
        The schema built with the rules given.
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


def get_config_generation_argname_value(
        args: argparse.Namespace) -> tuple[str, str]:
    """
    Get configuration type and value in the case of generating configurations,
    which should have been specified while parsing arguments.

    Args:
        args: Arguments parsed.

    Returns:
        Configuration type.
        Value of configuration type argument.
    """
    regex = r"generate_[a-z]+_config"
    arg_specified = [
        x for x in dir(args) if re.match(regex, x) and getattr(args, x)][0]
    config_type = arg_specified.split("_")[1]
    arg_value = getattr(args, arg_specified)
    return config_type, arg_value


class FileExtension:
    """
    Defines the available file formats to read data.
    """
    csv = '.csv'
    excel = ['.xlsx', '.xls']
    # Adding more file extensions here


class ProblemType:
    """
    Defines the available problem types.
    """
    classification = 'classification'
    regression = 'regression'
