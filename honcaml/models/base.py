from abc import ABC, abstractmethod
from typing import Dict, List, Callable

from honcaml.exceptions import model as model_exceptions
from honcaml.tools import custom_typing as ct
from honcaml.tools import utils


class BaseModel(ABC):
    """
    Model base class.

    Attributes:
        _estimator_type (str): The kind of estimator to be used. Valid values
            are `regressor` and `classifier`.
        _estimator: Estimator defined by child classes.
    """

    def __init__(self, estimator_type: str) -> None:
        """
        Base class constructor. Initializes the common attributes.

        Args:
            estimator_type: The kind of estimator to be used. Valid
                values are `regressor` and `classifier`.
        """
        if estimator_type not in estimator_types:
            raise model_exceptions.EstimatorTypeNotAllowed(estimator_type)
        self._estimator_type = estimator_type
        self._estimator = None

    @staticmethod
    def _import_model_library(model_config: dict) -> Callable:
        return utils.import_library(
            model_config['module'], model_config['hyperparameters'])

    @abstractmethod
    def read(self, settings: Dict) -> None:
        """
        Reads an estimator from disk. Must be implemented by child classes.

        Args:
            settings: Parameters configuring read operation.
        """
        pass

    @abstractmethod
    def build_model(self, model_config: Dict, normalizations: Dict) -> None:
        """
        Creates the requested estimator. Must be implemented by child classes.

        Args:
            model_config: Model configuration, i.e. module and its
                hyperparameters.
            normalizations: Definition of normalizations that applies to
                the dataset during the model pipeline.
        """
        pass

    @abstractmethod
    def fit(self, x: ct.Dataset, y: ct.Dataset, **kwargs: Dict) -> None:
        """
        Trains the estimator on the specified dataset. Must be implemented by
        child classes.

        Args:
            x: Dataset features.
            y: Dataset target.
            **kwargs: Extra parameters.
        """
        pass

    @abstractmethod
    def predict(self, x: ct.Dataset, **kwargs: Dict) -> List:
        """
        Uses the estimator to make predictions on the given dataset features.
        Must be implemented by child classes.

        Args:
            x: Dataset features.
            **kwargs: Extra parameters.

        Returns:
            Resulting predictions from the estimator.
        """
        pass

    @abstractmethod
    def evaluate(self, x: ct.Dataset, y: ct.Dataset, **kwargs: Dict) -> Dict:
        """
        Evaluates the estimator on the given dataset. Must be implemented by
        child classes.

        Args:
            x: Dataset features.
            y: Dataset target.
            **kwargs: Extra parameters.

        Returns:
            Resulting metrics from the evaluation.
        """
        pass

    @abstractmethod
    def save(self, settings: Dict) -> None:
        """
        Stores the estimator to disk. Must be implemented by child classes.

        Args:
            settings: Parameter settings defining the store operation.
        """
        pass


class ModelType:
    """
    Defines the available types of models.
    """
    sklearn = 'sklearn'


class EstimatorType:
    """
    Defines the available types of estimators.
    """
    classifier = 'classifier'
    regressor = 'regressor'


estimator_types = [
    EstimatorType.classifier,
    EstimatorType.regressor
]
