from abc import ABC, abstractmethod
from typing import Dict
import copy
from src.tools import utils
from src.tools.startup import logger


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
        self._step_settings = self._merge_settings(
            default_settings.copy(), user_settings.copy())

        # Check if it runs the parent method or child method
        self._validate_step()

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

    @property
    def extract_settings(self) -> Dict:
        """
        This is a getter method. This function returns the '_extract_settings'
        attribute.

        Returns:
            (str): a dict with settings of extract phase
        """
        return self._extract_settings

    @property
    def transform_settings(self) -> Dict:
        """
        This is a getter method. This function returns the
        '_transform_settings' attribute.

        Returns:
            (str): a dict with settings of transform phase
        """
        return self._transform_settings

    @property
    def load_settings(self) -> Dict:
        """
        This is a getter method. This function returns the '_load_settings'
        attribute.

        Returns:
            (str): a dict with settings of load phase
        """
        return self.load_settings

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
    def _extract(self, settings: Dict) -> None:
        """
        The extract process from the step ETL. This function must be
        implemented by child classes.

        Args:
            settings (Dict): the settings defining the extract ETL process.
        """
        pass

    @abstractmethod
    def _transform(self, settings: Dict) -> None:
        """
        The transform process from the step ETL. This function must be
        implemented by child classes.

        Args:
            settings (Dict): the settings defining the transform ETL process.
        """
        pass

    @abstractmethod
    def _load(self, settings: Dict) -> None:
        """
        The load process from the step ETL. This function must be
        implemented by child classes.

        Args:
            settings (Dict): the settings defining the load ETL process.
        """
        pass

    @abstractmethod
    def run(self, metadata: Dict) -> Dict:
        """
        Run the step.
        """
        pass

    def execute(self) -> None:
        """
        This function executes the ETL processes from the current step.
        This function runs the current steps.
        """
        if StepPhase.extract in self._step_settings:
            logger.info('Running extract phase...')
            self._extract(copy.deepcopy(self._extract_settings))
            logger.info('Extract phase complete.')
        if StepPhase.transform in self._step_settings:
            logger.info('Running transform phase...')
            self._transform(copy.deepcopy(self._transform_settings))
            logger.info('Transform phase complete.')
        if StepPhase.load in self._step_settings:
            logger.info('Running load phase...')
            self._load(copy.deepcopy(self._load_settings))
            logger.info('Load phase complete.')


class StepType:
    """
    This class defines the valid types of steps. The valid steps are the
    following:
        - data
        - model
        - benchmark
    """
    data = 'data'
    model = 'model'
    benchmark = 'benchmark'


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
