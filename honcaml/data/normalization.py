from typing import Dict, List, Tuple, Callable

from honcaml.tools import custom_typing as ct
from honcaml.tools import utils


class Normalization:
    """
    The aim of this class is to store the normalization parameters for dataset
    features and target.

    Attributes:
        _features (List[str]): list of columns to normalize.
        _target (List[str]): list of targets to normalize.
        _features_normalizer (Dict): normalization module and parameters to
            apply to a list of features.
        _target_normalizer (Dict): normalization module and parameters to
            apply to a list of targets.
    """

    def __init__(self, settings: Dict) -> None:
        """
        This is a constructor method of class. Given a settings dict, this
        function process the values of settings ans stores it to a class
        attributes.

        Args:
            settings (Dict): a dict with normalization configuration.
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
        This is a getter method. This function returns the list of features
        to normalize.

        Returns:
            (List[str]): a list of features.
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
        This is a getter method. This function returns the list of targets
        to normalize.

        Returns:
            (List[str]): a list of targets.
        """
        return self._target

    @property
    def target_normalizer(self) -> Callable:
        """
        This is a getter method. This function returns a tuple with the
        normalization module and parameters to apply to a target.

        Returns:
            (Tuple[str, dict]): a module and parameters for target.
        """
        module, params = self._get_module_and_params(self._target_normalizer)
        return utils.import_library(module, params)

    @staticmethod
    def _get_columns(settings: Dict, key: str) -> List[str]:
        """
        Given a settings as dict and key, this function gets the list of
        columns if key exists.

        Args:
            settings (Dict): normalization settings as dict.
            key (str): a key to get columns list. The possible values are
                'features' or 'target'.

        Returns:
            (List[str]): list of columns if the key exists. Otherwise, it gets
                empty list.
        """
        return settings.get(key, {}).pop('columns', [])

    @staticmethod
    def _get_module_and_parameters(settings: Dict, key: str) -> Dict:
        """
        Given a settings as dict and key, this function gets the all values
        from key. The values of dict could be 'module' and 'module_parameters.

        Args:
            settings (Dict): normalization settings as dict.
            key (str): a key to get values. The possible values are
                'features' or 'target'.

        Returns:
            (Dict): a dict values if the key exists. Otherwise, it gets
                empty dict.
        """
        return settings.pop(key, {})
