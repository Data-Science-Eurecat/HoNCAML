import importlib


def retrieve_problem_class(framework: str, problem_type: str) -> object:
    """
    Retrieve class required by framework and problem type.

    Args:
        framework: Framework name.
        problem_type: Problem type (regression, classification)

    Returns:
        Class instance required.
    """
    module_name = '.'.join(['src', 'frameworks', framework])
    module = importlib.import_module(module_name)
    class_name = framework.capitalize() + problem_type.capitalize()
    class_instance = getattr(module, class_name)()
    return class_instance
