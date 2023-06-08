from typing import Dict, List, Tuple, Callable

from honcaml.tools import custom_typing as ct
from honcaml.tools import utils


class Normalization:
    """
    The aim of this class is to store the normalization parameters for dataset
    features and target.

    Attributes:
        _features (List[str]): Columns to normalize.
        _target (List[str]): Targets to normalize.
        _features_normalizer (Dict): Normalization module and parameters to
            apply to a list of features.
        _target_normalizer (Dict): Normalization module and parameters to
            apply to a list of targets.
    """

    def __init__(self, settings: Dict) -> None:
        """
        Constructor method of class. Given a settings dict, this function
        process the values of settings and stores it to a class attributes.

        Args:
            settings: Normalization configuration.
        """
        # Features
        self._features: ct.StrList = self._get_columns(settings, 'features')
        self._features_normalizer: Dict = self._get_module_and_parameters(
            settings, 'features')

        # Target
        self._target: ct.StrList = self._get_columns(settings, 'target')
        self._target_normalizer: Dict = self._get_module_and_parameters(
            settings, 'target')

    @property
    def features(self) -> ct.StrList:
        """
        Getter method for '_features' attribute.

        Returns:
            '_features' current value.
        """
        return self._features

    @staticmethod
    def _get_module_and_params(norm_dict: dict) -> Tuple[str, dict]:
        """
        Given a normalization dict of features or target, this function gets
        the 'module' and 'module_parameters'.

        Args:
            norm_dict (dict): a dict that contains the normalization
                configuration.

        Returns:
            (Tuple[str, dict]): a module and parameters normalization.
        """
        return norm_dict['module'], norm_dict['module_params']

    @property
    def features_normalizer(self) -> Callable:
        """
        This is a getter method. This function returns a tuple with the
        normalization module and parameters to apply to a features.

        Returns:
            (Tuple[str, dict]): a module and parameters for features.
        """
        module, params = self._get_module_and_params(self._features_normalizer)
        return utils.import_library(module, params)

    @property
    def target(self) -> ct.StrList:
        """
        Getter method for '_target' attribute.

        Returns:
            '_target' current value.
        """
        return self._target

    @property
    def target_normalizer(self) -> Callable:
        """
        Getter method for '_target_normalizer' attribute.

        Returns:
            '_target_normalizer' current value.
        """
        module, params = self._get_module_and_params(self._target_normalizer)
        return utils.import_library(module, params)

    @staticmethod
    def _get_columns(settings: Dict, key: str) -> List[str]:
        """
        Get the list of columns for key in settings.

        Args:
            settings: Normalization settings as dict.
            key: A key to get columns list. The possible values are 'features'
                or 'target'.

        Returns:
            Columns if the key exists. Otherwise, it gets an empty list.
        """
        return settings.get(key, {}).pop('columns', [])

    @staticmethod
    def _get_module_and_parameters(settings: Dict, key: str) -> Dict:
        """
        Get all values from key in settings, which correspond to module and
        parameters.

        Args:
            settings: Normalization settings.
            key: Key to get values from. Possible values are 'features' or
                'target'.

        Returns:
            Key value if the key exists. Otherwise, it gets empty dict.
        """
        return settings.pop(key, {})
