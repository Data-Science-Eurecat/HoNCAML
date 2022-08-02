from src.models.model import Model
from src.data import extract
from src.data import load
from src.tools import utils
from typing import Dict


class RegressorModel(Model):
    """
    The regressor kind of model. The model that applies to regression problems.

    Attributes:
        action_settings (Dic,t): the parameters that define each action from 
        the ETL process.
        model (...): the model defined by a library (s.t. sklearn).
    """

    def __init__(self) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters for this specific model.
        """
        self.model = None

    def read(self, settings: Dict):
        """
        ETL model extract. Reads a previously saved regressor model.
        """
        self.model = extract.read_model(settings)

    def build_model(self, settings: Dict):
        self.model = utils.import_library(
            settings['library'], settings['hyperparameters'])

    def train(self, settings: Dict) -> None:
        if self.model is None:
            self.model.build_model(objects['model'])
        pass

    def cross_validate(self, settings: Dict) -> None:
        pass

    def save(self, settings: Dict) -> None:
        """
        ETL model load. Save the regressor model into disk. 
        """
        load.save_model(self.model, settings)
