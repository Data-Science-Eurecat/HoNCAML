from abc import ABC, abstractmethod
from typing import Dict, List
from src.tools import custom_typing as ct


class BaseModel(ABC):
    """
    Model base class.

    Attributes:
        _estimator_type (str): the kind of estimator to be used. Valid values
            are `regressor` and `classifier`.
        _estimator (TODO type): an estimator defined by child classes.
    """

    def __init__(self, estimator_type: str) -> None:
        """
        Base class constructor. Initializes the common attributes.

        Args:
            estimator_type (str): the kind of estimator to be used. Valid values
                are `regressor` and `classifier`.
        """
        self._estimator_type = estimator_type
        self._estimator = None

    @abstractmethod
    def read(self, settings: Dict) -> None:
        """
        Read an estimator from disk. This function must be implemented by
        child classes.

        Args:
            settings (Dict): the parameter settings defining the read 
                operation.
        """
        pass

    @abstractmethod
    def build_model(self, model_config: Dict, normalizations: Dict) -> None:
        """
        Create the requested estimator. This function must be implemented by
        child classes.

        Args:
            model_config (Dict): the model configuration: the module and their
                hyperparameters.
            normalizations (Dict): the definition of normalizations applied to
                the dataset during the model pipeline.
        """
        pass

    @abstractmethod
    def fit(self, x: ct.Dataset, y: ct.Dataset, **kwargs: Dict) -> None:
        """
        Train the estimator on the specified dataset. This function must be
        implemented by child classes.

        Args:
            x (ct.Dataset): the dataset features.
            y (ct.Dataset): the dataset target.
            **kwargs (Dict): extra parameters.
        """
        pass

    @abstractmethod
    def predict(self, x: ct.Dataset, **kwargs: Dict) -> List:
        """
        Use the estimator to make predictions on the given dataset features.
        This function must be implemented by child classes.

        Args:
            x (ct.Dataset): the dataset features.
            **kwargs (Dict): extra parameters.

        Returns:
            predictions (List): the resulting predictions from the estimator.
        """
        pass

    @abstractmethod
    def evaluate(self, x: ct.Dataset, y: ct.Dataset, **kwargs: Dict) -> Dict:
        """
        Evaluate the estimator on the given dataset. This function must be
        implemented by child classes.

        Args:
            x (ct.Dataset): the dataset features.
            y (ct.Dataset): the dataset target.
            **kwargs (Dict): extra parameters.

        Returns:
            metrics (Dict): the resulting metrics from the evaluation.
        """
        pass

    @abstractmethod
    def save(self, settings: Dict) -> None:
        """
        Store the estimator to disk. This function must be implemented by
        child classes.

        Args:
            settings (Dict): the parameter settings defining the store
                operation.
        """
        pass


class ModelType:
    """
    This class defines the available types of models. The valid steps are the
    following:
        - sklearn
    """
    sklearn = 'sklearn'
