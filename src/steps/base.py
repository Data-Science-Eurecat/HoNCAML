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
    Abstract class Step to wrap a pipeline step. It defines the base structure
    for an step from the main pipeline.

    Attributes:
        step_settings (Dict): the settings that define the step.
        extract_settings (Dict): the settings defining the extract ETL process.
        transform_settings (Dict): the settings defining the transform ETL process.
        load_settings (Dict): the settings defining the load ETL process.
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
        """
        Merge the user defined settings with the default ones.

        Args:
            default_settings (Dict): the default settings for the step.
            default_settings (Dict): the user defined settings for the step.

        Returns:
            merged_settings (Dict): the user and default settings merged.
        """
        pass

    @abstractmethod
    def validate_step(self) -> None:
        """
        Validates the settings for the step ensuring that the step has the
        mandatory keys to run.
        """
        pass

    @abstractmethod
    def extract(self) -> None:
        """
        The extract process from the step ETL. This function must be
        implemented by child classes.
        """
        pass

    @abstractmethod
    def transform(self) -> None:
        """
        The transform process from the step ETL. This function must be
        implemented by child classes.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        The load process from the step ETL. This function must be
        implemented by child classes.
        """
        pass

    def execute(self) -> None:
        """
        This function executes the ETL processes from the current step.

        Args:
            objects (Dict): the objects output from each different previous
                step.

        Returns:
            objects (Dict): the previous objects updated with the ones from
                the current step.
        """
        if StepProcesses.extract in self.step_settings:
            self.extract()
        if StepProcesses.transform in self.step_settings:
            self.transform()
        if StepProcesses.load in self.step_settings:
            self.load()
