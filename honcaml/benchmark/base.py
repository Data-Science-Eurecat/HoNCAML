from abc import ABC, abstractmethod
from ray import tune

from honcaml.exceptions import benchmark as exceptions


BENCHMARK_TYPES = ['sklearn', 'torch']
BENCHMARK_OPTIONS = {'torch': 'layers'}


class BaseBenchmark(ABC):

    def __init__(self, name: str) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current steps.

        Args:
            name: Name of model type, which sets type of benchmark.
        """

    @abstractmethod
    def _clean_search_space(search_space: dict) -> dict:
        """
        Given a dict with a search space for a model, this function gets the
        module of model to import and the hyperparameters search space and
        ensures that method exists.

        Must be implemented by child classes.

        Args:
            search_space (Dict): a dict with the search space to explore

        Returns:
            (Dict): a dict where for each hyperparameter the corresponding
            method to generate all possible values during the search.
        """
        pass

    @classmethod
    def _clean_parameter_search_space(
            cls, name: str, space: dict) -> dict:
        """
        Convert parameter search space to tune formatting in order to be
        correctly fed afterwards.

        Args:
            space: Search space for parameter.

        Returns:
            Correctly formatted space.

        Raises:
            Exception in case that tune method does not exist.
        """
        search_space = {}
        # Append internal spaces if there are any
        internal_spaces = cls._clean_internal_params_for_search_space(
            name, space)
        if internal_spaces:
            search_space.update(internal_spaces)
            # Prune internal spaces
            for i, option in enumerate(space['values']):
                space['values'][i].__delitem__('params')
        # Append main parameter
        try:
            tune_method = getattr(tune, space['method'])
        except AttributeError:
            raise exceptions.TuneMethodDoesNotExists(space['method'])
        if space['method'] in ['choice', 'grid_search']:
            space['values'] = [space['values']]
        search_space.update({name: tune_method(*space['values'])})
        return search_space

    @classmethod
    def _clean_internal_params_for_search_space(
            cls, name: str, space: dict,
            format_parts: str = '[{}]', join_parts: str = '-') -> dict:
        """
        Cleans internal parameters to consider for search space; this checks if
        there are nested parameters for the benchmark a part from the main one,
        and returns them in a specific format that will be handled in the
        `EstimatorTrainer._parse_param_space` method properly.

        Args:
            name: Name of main parameter that is being handled
            space: Search space for parameter
            format_parent: Format in which parent will be

        Returns:
            Dictionary with all internal parameters.
        """
        internal_spaces = {}
        if space['method'] == 'choice' and isinstance(space['values'], list):
            for i, element in enumerate(space['values']):
                if isinstance(element, dict) and 'params' in element:
                    for param_name, param_val in element['params'].items():
                        if 'method' and 'values' in param_val:
                            prefix_main = format_parts.format(name)
                            prefix_module = format_parts.format(
                                element['module'])
                            internal_name = join_parts.join(
                                [prefix_main, prefix_module, param_name])
                            internal_val = cls._clean_parameter_search_space(
                                param_name, param_val)
                            internal_spaces.update(
                                {internal_name: internal_val[param_name]})
        return internal_spaces
