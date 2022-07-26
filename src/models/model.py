from abc import ABC, abstractmethod


class Model(ABC):
    # TODO: parent class, create new ones e.g RegressorModel, ClassifierModel
    pass

    def __init__():
        pass

    @abstractmethod
    def extract(self):
        """
        ETL model extract. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def transform(self):
        """
        ETL model transform. This function must be implemented by child classes.
        """
        pass

    @abstractmethod
    def load(self):
        """
        ETL model load. This function must be implemented by child classes.
        """
        pass
