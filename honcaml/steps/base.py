from abc import ABC, abstractmethod
from typing import Dict
import copy
from honcaml.tools import utils
from honcaml.tools.startup import logger
from honcaml.exceptions import step as step_exception


class BaseStep(ABC):
    """
    Abstract class to wrap a pipeline step. It defines the base structure
    for a step from the main pipeline.

    Attributes:
        _step_settings (Dict): Settings that define the step.
        _extract_settings (Dict): Settings defining the extract ETL
            process.
        _transform_settings (Dict): Settings defining the transform ETL
            process.
        _load_settings (Dict): Settings defining the load ETL process.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict,
                 step_rules: Dict) -> None:
        """
        Constructor method of class. It initializes the common steps
        parameters.

        Args:
            default_settings: Default settings for the steps.
            user_settings: User-defined settings for the steps.
        """
        self._step_rules = step_rules
        self._step_settings = self._merge_settings(
            default_settings.copy(), user_settings.copy())

        # TODO: review and fix step validation
        # self._validate_step()

        self._extract_settings = \
            self._step_settings.get(StepPhase.extract, None)
        self._transform_settings = \
            self._step_settings.get(StepPhase.transform, None)
        self._load_settings = \
            self._step_settings.get(StepPhase.load, None)

    @property
    def step_settings(self) -> Dict:
        """
        Getter method for the '_step_settings' attribute.

        Returns:
            '_step_settings' current value.
        """
        return self._step_settings

    @property
    def extract_settings(self) -> Dict:
        """
        Getter method for the '_extract_settings' attribute.

        Returns:
            '_extract_settings' current value.
        """
        return self._extract_settings

    @property
    def transform_settings(self) -> Dict:
        """
        Getter method for the '_transform_settings' attribute.

        Returns:
            '_transform_settings' current value.
        """
        return self._transform_settings

    @property
    def load_settings(self) -> Dict:
        """
        Getter method for the '_load_settings' attribute.

        Returns:
            '_load_settings' current value.
        """
        return self.load_settings

    def __str__(self):
        return str(self._step_settings)

    def __repr__(self):
        return str(self._step_settings)

    def _merge_settings(
            self, default_settings: Dict, user_settings: Dict) -> Dict:
        """
        Merge two defined settings; the first one considered the default,
        whereas the second is considered the user ones. In case of conflict in
        keys, user-defined settings prevail.

        Args:
            default_settings: Default configuration.
            user_settings: User custom configuration.

        Returns:
            Merged configuration.
        """
        step_settings = {}
        for phase in step_phases:
            if phase in user_settings:
                # Getting params of phase
                phase_default_settings = default_settings.get(phase, {})
                phase_user_settings = user_settings.get(phase, {})

                # Combine default settings and user settings
                phase_settings = utils.update_dict_from_default_dict(
                    phase_default_settings, phase_user_settings)

                step_settings[phase] = phase_settings

        return step_settings

    def _validate_step(self) -> None:
        """
        Validates settings for the step, ensuring that the step has the
        mandatory keys to run.
        """
        validator = utils.build_validator(self._step_rules)
        if not validator.validate(self._step_settings):
            raise step_exception.StepValidationError(validator.errors)

    @abstractmethod
    def _extract(self, settings: Dict) -> None:
        """
        Extract process from the step ETL. Must be implemented by child
        classes.

        Args:
            settings: Settings defining the extract ETL process.
        """
        pass

    @abstractmethod
    def _transform(self, settings: Dict) -> None:
        """
        Transform process from the step ETL. Must be implemented by child
        classes.

        Args:
            settings: Settings defining the transform ETL process.
        """
        pass

    @abstractmethod
    def _load(self, settings: Dict) -> None:
        """
        Load process from the step ETL. Must be implemented by child
        classes.

        Args:
            settings: Settings defining the load ETL process.
        """
        pass

    @abstractmethod
    def run(self, metadata: Dict) -> Dict:
        """
        Runs the step.

        Args:
            metadata: Configuration parameters in order to run the step.
        """
        pass

    def execute(self) -> None:
        """
        Executes the ETL processes from the current step.
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
    Defines the valid types of steps. Valid steps are the following:
    - data
    - model
    - benchmark
    """
    data = 'data'
    model = 'model'
    benchmark = 'benchmark'


class StepPhase:
    """
    Defines available valid step phases.
    """
    extract = 'extract'
    transform = 'transform'
    load = 'load'


# List of phases for each step
step_phases = [
    StepPhase.extract,
    StepPhase.transform,
    StepPhase.load]
