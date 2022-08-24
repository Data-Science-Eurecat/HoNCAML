from abc import ABC, abstractmethod
from typing import Dict


class BaseDataset(ABC):
    """
    Base class defining a dataset.

    Attributes:
        action_settings (Dict): the parameters that define each action from 
        the ETL process.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        This is a constructor method of class. This function initializes
        the common parameters for a dataset.
        """
        pass

    @abstractmethod
    def read(self, settings: Dict):
        """
        ETL data extract. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def preprocess(self, settings: Dict):
        """
        ETL data transform. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def save(self, settings: Dict):
        """
        ETL data load. This function must be implemented by child classes.
        """
        pass
