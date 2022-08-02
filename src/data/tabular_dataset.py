from src.data.dataset import Dataset
from src.data import extract
from src.data import transform
from src.data import load
from typing import Dict
import pandas as pd


class TabularDataset(Dataset):
    """
    A dataset consisting in tabular data. The data read come from files
    encoding the data as tables.

    Attributes:
        action_settings (Dict): the parameters that define each action from 
        the ETL process.
        dataset (pd.DataFrame): the dataframe read from the tabular file data.
        target (pd.DataFrame): the dataset target column.
    """

    def __init__(self) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters for this specific dataset.
        """
        self.dataset = None
        self.target = None

    def read(self, settings: Dict):
        """
        ETL data extract. Reads data from a file that encodes the data as
        tables (e.g. excel, csv).
        """
        dataset = extract.read_dataframe(settings)
        self.dataset = dataset.drop(settings['target'], axis=1)
        self.target = dataset[[settings['target']]]

    def preprocess(self, settings: Dict):
        """
        ETL data transform. Apply the transformations requested to the data.
        """
        self.dataset, self.target = transform.process_data(
            self.dataset, self.target, settings)

    def save(self, settings: Dict):
        """
        ETL data load. Save the dataset into disk.
        """
        dataset = pd.concat((self.dataset, self.target), axis=1)
        load.save_dataframe(dataset, settings)
