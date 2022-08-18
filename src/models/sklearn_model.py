from src.models import base
from typing import Dict, List
from src.tools import utils


class SklearnModel(base.BaseModel):
    """
    Model base class.

    Attributes:

    """

    def __init__(self, estimator_type: str) -> None:
        super().__init__(estimator_type)

    def read(self, settings: Dict) -> None:
        """
        ETL model read. This function must be implemented by child classes.
        name format: estimator_type.unique_id
        """
        pass

    def build_model(self):
        pass

    def fit(self, X, y, **kwargs) -> None:
        """
        ETL model fit. This function must be implemented by child classes.
        """
        self.estimator = self.estimator.fit(X, y)

    def evaluate(self, settings: Dict) -> Dict:
        """
        ETL model evaluate. This function must be implemented by child classes.
        """
        pass

    def predict(self, settings: Dict) -> List:
        """
        ETL model predict. This function must be implemented by child classes.
        """
        pass

    def save(self, settings: Dict) -> None:
        """
        ETL model save. This function must be implemented by child classes.
        format: estimator_type.unique_id
        """
        filename = utils.generate_unique_id(
            self.estimator_type, adding_uuid=True)
        pass
