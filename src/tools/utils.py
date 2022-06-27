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
