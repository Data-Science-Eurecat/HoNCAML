from abc import ABC, abstractmethod
from typing import Dict
from src.tools import utils


class BaseStep(ABC):
    """
    Abstract class Step to wrap the pipeline's steps. Defines the base 
    structure for an steps from the main pipeline.

    Attributes:
        step_settings (Dict): the settings that define the steps.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the common steps parameters.

        Args:
            default_settings (Dict): the default settings for the steps.
            user_settings (Dict): the user defined settings for the steps.
        """
        # Check if it runs the parent method or child method
        self.validate_step()

        self.step_settings = self._merge_settings(
            default_settings.copy(), user_settings.copy())

        self.extract_settings = \
            self.step_settings.get(StepPhase.extract, None)
        self.transform_settings = \
            self.step_settings.get(StepPhase.transform, None)
        self.load_settings = \
            self.step_settings.get(StepPhase.load, None)

    def __str__(self):
        return self.step_settings

    def __repr__(self):
        return self.step_settings

    def _merge_settings(
            self, default_settings: Dict, user_settings: Dict) -> Dict:
        """
        Given two dictionaries first one with a default settings and the
        second one with the user settings, this function combine user settings
        and default settings. In addition, the params of user setting prevails.

        Args:
            default_settings (Dict): a dict with default settings.
            user_settings (Dict): a dict with user custom settings.

        Returns:
            a dict with pipeline settings.
        """
        step_settings = {}
        for phase in step_phases:
            # Getting params of phase
            phase_default_settings = default_settings.get(phase, {})
            phase_user_settings = user_settings.get(phase, {})

            # Combine default settings and user settings
            phase_settings = utils.update_dict_from_default_dict(
                phase_default_settings, phase_user_settings)

            if phase_settings:
                step_settings[phase] = phase_settings

        return step_settings

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

    def execute(self, objects: Dict) -> None:
        """
        This function runs the current steps.

        Args:
            objects (Dict): the objects output from each different previous
                steps.

        Returns:
            objects (Dict): the previous objects updated with the ones from
                the current steps.
        """
        if StepPhase.extract in self.step_settings:
            self.extract()
        if StepPhase.transform in self.step_settings:
            self.transform()
        if StepPhase.load in self.step_settings:
            self.load()


class StepType:
    """
    This class defines the valid types of steps. The valid steps are the
    following:
        - data
        - model
    """
    data = 'data'
    model = 'model'


class StepPhase:
    """
    Class with the aim to store processes from a steps (the ETL ones).
    """
    extract = 'extract'
    transform = 'transform'
    load = 'load'


# List of phases for each step
step_phases = [
    StepPhase.extract,
    StepPhase.transform,
    StepPhase.load]
