from src.models.model import Model
from src.models import extract as extract_model
from src.models import transform as transform_model
from src.models import load as load_model
from typing import Dict


class RegressorModel(Model):
    """
    The regressor kind of model. The model that applies to regression problems.

    Attributes:
        action_settings (Dic,t): the parameters that define each action from 
        the ETL process.
        model (...): the model defined by a library (s.t. sklearn).
    """

    def __init__(self, action_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters for this specific model.

        Args:
            action_settings (Dict): the parameters that define each action 
            from the ETL process.
        """
        super().__init__(action_settings)
        self.model = None

    def extract(self):
        """
        ETL model extract. Reads a previously saved regressor model.
        """
        extract_settings = self.action_settings.get('extract')
        if extract_settings is not None:
            self.model = extract_model.read_model(extract_settings)
        else:
            # TODO: init model
            pass

    def transform(self):
        """
        ETL model transform. Perform the transformations requested.
        """
        transform_settings = self.action_settings.get('transform')
        if transform_settings is not None:
            self.model = transform_model.transform(  # TODO: change and implement method
                self.model, transform_settings)

    def load(self):
        """
        ETL model load. Save the regressor model into disk. 
        """
        load_settings = self.action_settings.get('load')
        if load_settings is not None:
            load_model.save_model(self.model, load_settings)
