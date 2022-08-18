from abc import ABC, abstractmethod
from typing import Dict, List


class BaseModel(ABC):
    """
    Model base class.

    Attributes:

    """

    def __init__(self, estimator_type: str) -> None:
        # self.model_config = None
        self.estimator_type = estimator_type
        self.estimator = None

    @abstractmethod
    def read(self, settings: Dict) -> None:
        """
        ETL model read. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs) -> None:
        """
        ETL model fit. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def evaluate(self, settings: Dict) -> Dict:
        """
        ETL model evaluate. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def predict(self, settings: Dict) -> List:
        """
        ETL model predict. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def save(self, settings: Dict) -> None:
        """
        ETL model save. This function must be implemented by child classes.
        """
        pass
