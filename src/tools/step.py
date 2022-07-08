from abc import ABC, abstractmethod
from typing import Dict


class Step(ABC):
    """
    Abstract class Step. Defines the base structure for an step from the main
    pipeline.

    Attributes:
        default_settings (Dict): the default settings for the step.
        user_settings (Dict): the user defined settings for the step.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the common step parameters.

        Args:
            default_settings (Dict): the default settings for the step.
            user_settings (Dict): the user defined settings for the step.
        """
        self.default_settings = default_settings
        self.user_settings = user_settings

    @abstractmethod
    def _setup(self) -> None:
        """
        The function to setup the specific step.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        This function runs the current step.
        """
        pass
