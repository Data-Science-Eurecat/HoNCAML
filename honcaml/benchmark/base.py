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

    @staticmethod
    def _clean_parameter_search_space(space: dict) -> callable:
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
        try:
            tune_method = getattr(tune, space['method'])
        except AttributeError:
            raise exceptions.TuneMethodDoesNotExists(space['method'])
        if space['method'] in ['choice', 'grid_search']:
            space['values'] = [space['values']]
        search_space = tune_method(*space['values'])
        return search_space
