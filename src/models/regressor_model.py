from src.models.model import Model
from src.data import extract
from src.data import load
from src.tools import utils
from models import general
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
        super().__init__()

    def read(self, settings: Dict):
        """
        ETL model extract. Reads a previously saved regressor model.
        """
        self.model = extract.read_model(settings)

    def build_model(self, settings: Dict):
        self.model = utils.import_library(
            settings['library'], settings['hyperparameters'])

    def train(self, settings: Dict) -> None:
        X, y = self.dataset.get_data()
        X = X[[settings['features']]]
        self.model.fit(X, y)

    def cross_validate(self, settings: Dict) -> Dict:
        cv_results = []
        for fold in settings['cv_folds']:
            X_train, X_test, y_train, y_test = self.dataset.train_test_split(
                settings['validation_split'], fold)
            # TODO: reset model
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            cv_results.append(general.compute_metrics(
                y_test, y_pred, settings['metrics']))
        results = general.aggregate_cv_results(cv_results)
        results = dict(zip(settings['metrics'], results))
        return results

    def evaluate(self, settings: Dict) -> Dict:
        X, y = self.dataset.get_data()
        y_pred = self.model.predict(X)
        results = general.compute_metrics(y, y_pred, settings['metrics'])
        results = dict(zip(settings['metrics'], results))
        return results

    def predict(self, settings: Dict):
        X, y = self.dataset.get_data()
        y_pred = self.model.predict(X)
        return y_pred

    def save(self, settings: Dict) -> None:
        """
        ETL model load. Save the regressor model into disk. 
        """
        load.save_model(self.model, settings)
