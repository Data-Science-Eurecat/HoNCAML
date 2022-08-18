from abc import ABC, abstractmethod
from typing import Dict


class StepProcesses:
    """
    Class with the aim to store processes from a step (the ETL ones).
    """
    extract = 'extract'
    transform = 'transform'
    load = 'load'


class BaseStep(ABC):
    """
    Abstract class Step to wrap the pipeline's steps. Defines the base 
    structure for an step from the main pipeline.

    Attributes:
        step_settings (Dict): the settings that define the step.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the common step parameters.

        Args:
            default_settings (Dict): the default settings for the step.
            user_settings (Dict): the user defined settings for the step.
        """
        # Check if it runs the parent method or child method
        self.validate_step()

        self.step_settings = self._merge_settings(
            default_settings, user_settings)

        self.extract_settings = \
            self.step_settings.get(StepProcesses.extract, None)
        self.transform_settings = \
            self.step_settings.get(StepProcesses.transform, None)
        self.load_settings = \
            self.step_settings.get(StepProcesses.load, None)

    @abstractmethod
    def _merge_settings(
            self, default_settings: Dict, user_settings: Dict) -> Dict:
        pass

    @abstractmethod
    def validate_step(self):
        pass

    @abstractmethod
    def extract(self) -> None:
        pass

    @abstractmethod
    def transform(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    def execute(self) -> None:
        """
        This function runs the current step.

        Args:
            objects (Dict): the objects output from each different previous
                step.

        Returns:
            objects (Dict): the previous objects updated with the ones from
                the current step.
        """
        if StepProcesses.extract in self.step_settings:
            self.extract()
        else:
            self.build_model()
        if StepProcesses.transform in self.step_settings:
            self.transform()
        if StepProcesses.load in self.step_settings:
            self.load()
