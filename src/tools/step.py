from abc import ABC, abstractmethod


class Step(ABC):
    def __init__(self, settings):
        pass

    
    @abstractmethod
    def extract(self):
        pass

    
    @abstractmethod
    def transform(self):
        pass


    @abstractmethod
    def load(self):
        pass