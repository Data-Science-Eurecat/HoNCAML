from src.data.dataset import Dataset
from src.data import extract as extract_data
from src.data import transform as transform_data
from src.data import load as load_data
from typing import Dict


class TabularDataset(Dataset):
    """
    A dataset consisting in tabular data. The data read come from files
    encoding the data as tables.

    Attributes:
        action_settings (Dict): the parameters that define each action from 
        the ETL process.
        dataset (pd.DataFrame): the dataframe read from the tabular file data.
        target (str): the dataset target column.
    """

    def __init__(self, action_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters for this specific dataset.

        Args:
            action_settings (Dict): the parameters that define each action 
            from the ETL process.
        """
        super().__init__(action_settings)
        self.dataset = None
        self.target = None

    def extract(self):
        """
        ETL data extract. Reads data from a file that encodes the data as
        tables (e.g. excel, csv).
        """
        extract_settings = self.action_settings.get('extract')
        if extract_settings is not None:
            self.dataset, self.target = extract_data.read_data(
                extract_settings)

    def transform(self):
        """
        ETL data transform. Apply the transformations requested to the data.
        """
        transform_settings = self.action_settings.get('transform')
        if transform_settings is not None:
            self.dataset, self.target = transform_data.process_data(
                self.dataset, self.target, transform_settings)

    def load(self):
        """
        ETL data load. Save the dataset into disk.
        """
        load_settings = self.action_settings.get('load')
        if load_settings is not None:
            load_data.save_data(self.dataset, self.target, load_settings)
