from abc import ABC, abstractmethod
from typing import Dict, Union

from honcaml.data import normalization as norm


class BaseDataset(ABC):
    """
    Base class defining a dataset.

    Attributes:
        _normalization (Union[norm.Normalization, None]): class to store the
            normalization parameters for features and target.
    """

    def __init__(self) -> None:
        """
        This is a constructor method of class. This function initializes
        the common parameters for a dataset.
        """

        self._normalization: Union[norm.Normalization, None] = None

    @property
    def normalization(self) -> norm.Normalization:
        """
        This is a getter method. This function returns the '_normalization'
        attribute.

        Returns:
            (pd.DataFrame): dataset as pd.DataFrame
        """
        return self._normalization

    @normalization.setter
    def normalization(self, value: norm.Normalization) -> None:
        """
        This is a setter method. Given a Normalization instance, this function
        assigned it to a _normalization attribute.
        Args:
            value (norm.Normalization):

        """
        self._normalization = value

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
