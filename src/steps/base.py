from abc import ABC, abstractmethod
from typing import Dict

from src.tools import utils


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
    for a step from the main pipeline.

    Attributes:
        _step_settings (Dict): the settings that define the step.
        _extract_settings (Dict): the settings defining the extract ETL process.
        _transform_settings (Dict): the settings defining the transform ETL process.
        _load_settings (Dict): the settings defining the load ETL process.
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
        self._validate_step()

        self._step_settings = self._merge_settings(
            default_settings.copy(), user_settings.copy())

        self._extract_settings = \
            self._step_settings.get(StepPhase.extract, None)
        self._transform_settings = \
            self._step_settings.get(StepPhase.transform, None)
        self._load_settings = \
            self._step_settings.get(StepPhase.load, None)

    @property
    def step_settings(self) -> Dict:
        """
        This is a getter method. This function returns the '_step_settings'
        attribute.

        Returns:
            (str): a dict with settings of step
        """
        return self._step_settings

    def __str__(self):
        return str(self._step_settings)

    def __repr__(self):
        return str(self._step_settings)

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
            (Dict): pipeline settings as dict.
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
    def _validate_step(self) -> None:
        """
        Validates the settings for the step ensuring that the step has the
        mandatory keys to run.
        """
        pass

    @abstractmethod
    def _extract(self) -> None:
        """
        The extract process from the step ETL. This function must be
        implemented by child classes.
        """
        pass

    @abstractmethod
    def _transform(self) -> None:
        """
        The transform process from the step ETL. This function must be
        implemented by child classes.
        """
        pass

    @abstractmethod
    def _load(self) -> None:
        """
        The load process from the step ETL. This function must be
        implemented by child classes.
        """
        pass

    def execute(self) -> None:
        """
        This function executes the ETL processes from the current step.
        This function runs the current steps.
        """
        if StepPhase.extract in self._step_settings:
            self._extract()
        if StepPhase.transform in self._step_settings:
            self._transform()
        if StepPhase.load in self._step_settings:
            self._load()


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
