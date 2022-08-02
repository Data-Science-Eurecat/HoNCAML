from abc import ABC, abstractmethod
from typing import Dict


class Model(ABC):
    """
    Model base class.

    Attributes:

    """

    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def read(self, settings: Dict):
        """
        ETL model extract. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def build_model(self, settings: Dict):
        pass

    @abstractmethod
    def train(self, settings: Dict):
        """
        ETL model transform. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def evaluate(self, settings: Dict):
        """
        ETL model transform. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def load(self, settings: Dict):
        """
        ETL model load. This function must be implemented by child classes.
        """
        pass
