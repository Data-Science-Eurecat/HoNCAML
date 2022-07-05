from src import exceptions
from typing import Dict
import datetime
import uuid


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


def generate_unique_id(adding_uuid: bool = False) -> str:
    """
    This function generates a unique string id based on current timestamp and
    uuid4.

    Args:
        adding_uuid (optional, bool): adding uuid4 in generated id.

    Returns:
        a unique id string.
    """
    unique_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if adding_uuid:
        unique_id = f'{unique_id}_{uuid.uuid4()}'

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