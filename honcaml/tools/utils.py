import argparse
import datetime
import importlib
import re
import uuid
from cerberus import Validator
from functools import reduce
from typing import Dict, Callable, List


def import_library(
        module: str, params: Dict = None,
        mand_argument: object = None) -> Callable:
    """
    Imports the module specified and creates a new callable with specific
    parameters.

    Args:
        module: Module name.
        params: Parameters for the specific module initialization.
        mand_argument: Any mandatory argument to instantiate the library, which
            cannot be passed as params

    Returns:
        callable of imported module with parameters.
    """
    library = '.'.join(module.split('.')[:-1])
    name = module.split('.')[-1]
    imported_module = importlib.import_module(library)

    if params is None:
        params = dict()

    if mand_argument:
        inst = getattr(imported_module, name)(mand_argument, **params)
    else:
        inst = getattr(imported_module, name)(**params)
    return inst


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
        default_dict: Dict, source_dict: Dict, parent=None, grandparent=None,
        forbidden_parents: List = ['steps', 'models'],
        forbidden_keys: List = ['fit', 'predict', 'benchmark'],
        forbidden_grandparents: List = ['models']) -> Dict:
    """
    Combines two configurations prevailing the second over the default one.
    In addition, it throws a recursion over the dictionary values.
    In general, all keys not found are added to the dictionary, with the
    following exceptions:
    - Keys with parent any of forbidden_parents
    - Keys with name any of forbidden_keys

    Args:
        default_dict: Default configuration.
        source_dict: New configuration.
        parent: Parent key if there is any.
        grandparent: Grandparent key if there is any.
        forbidden_parents: If parent is any of forbidden, do not include key.
        forbidden_keys: If key is any of forbidden, do not include key.
        forbidden_grandparents: If grandparent is any of forbidden, do not
        include key.

    Returns:
        A configuration with merged values.
    """
    if source_dict is None:
        source_dict = {}
    for key, value in default_dict.items():
        if key not in source_dict:
            if value and parent not in forbidden_parents and \
               key not in forbidden_keys and \
                    grandparent not in forbidden_grandparents:
                source_dict[key] = value
        else:
            if isinstance(value, dict):
                source_dict[key] = update_dict_from_default_dict(
                    value, source_dict[key], key, parent)
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


def get_configuration_arguments(
        args: argparse.Namespace) -> tuple[str, str, str]:
    """
    Get configuration options in the case of generating configurations,
    which should have been specified while parsing arguments.

    Args:
        args: Arguments parsed.

    Returns:
        - User level
        - File path
        - Type of pipeline
    """
    regex = r"generate_[a-z]+_config"
    arg_specified = [
        x for x in dir(args) if re.match(regex, x) and getattr(args, x)][0]
    config_level = arg_specified.split("_")[1]
    filepath = getattr(args, arg_specified)
    pipeline_type = args.pipeline_type
    return config_level, filepath, pipeline_type


def select_scope_params(params: Dict, scope_list: List) -> Dict:
    """
    Select params that apply to the scope of the problem. For example, if
    there are default options for both regression and classification
    problem types, select the key corresponding to the pipeline problem
    type. Assuming scope (problem type) is regression, an example would be:

    Before: {'key': {'regression': 1, 'classification': 0}}
    After: {'key': 1}

    Args:
        params: Input parameters.
        scope_list: List of scopes to be selected.

    Returns:
        New parameters with scope applied.
    """
    if isinstance(params, dict):
        for key in params:
            if isinstance(params[key], dict):
                val_intersection = set(
                    list(params[key])).intersection(scope_list)
                if val_intersection:
                    val_to_assign = list(val_intersection)[0]
                    scope_params = params[key][val_to_assign]
                    params[key] = scope_params
                else:
                    scope_params = params[key]
                    select_scope_params(scope_params, scope_list)


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
