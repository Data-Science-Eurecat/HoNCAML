from abc import ABC, abstractmethod
from typing import Dict


class Dataset(ABC):
    """
    Base class defining a dataset.

    Attributes:
        action_settings (Dict): the parameters that define each action from 
        the ETL process.
    """

    def __init__(self, action_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the common parameters for a dataset.

        Args:
            action_settings (Dict): the parameters that define each action 
            from the ETL process.
        """
        self.action_settings = action_settings

    @abstractmethod
    def extract(self):
        """
        ETL data extract. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def transform(self):
        """
        ETL data transform. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def load(self):
        """
        ETL data load. This function must be implemented by child classes.
        """
        pass
