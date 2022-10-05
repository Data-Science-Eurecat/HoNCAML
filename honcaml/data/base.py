from abc import ABC, abstractmethod
from typing import Dict, Union

from honcaml.data import normalization as norm


class BaseDataset(ABC):
    """
    Base class defining a dataset.

    Attributes:
        _normalization (Union[norm.Normalization, None]): Class to store the
            normalization parameters for features and target.
    """

    def __init__(self) -> None:
        """
        Constructor method of class. It initializes the common parameters for a
        dataset.
        """
        self._normalization: Union[norm.Normalization, None] = None

    @property
    def normalization(self) -> norm.Normalization:
        """
        Getter method for '_normalization' attribute.

        Returns:
            '_normalization' current value.
        """
        return self._normalization

    @normalization.setter
    def normalization(self, value: norm.Normalization) -> None:
        """
        Setter method for '_normalization' attribute.

        Args:
            value: Value to assign to attribute.

        """
        self._normalization = value

    @abstractmethod
    def read(self, settings: Dict):
        """
        ETL data extract. Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def preprocess(self, settings: Dict):
        """
        ETL data transform. Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def save(self, settings: Dict):
        """
        ETL data load. Must be implemented by child classes.
        """
        pass
