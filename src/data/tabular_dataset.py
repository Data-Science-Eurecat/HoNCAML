from src.data.dataset import Dataset
from src.data import extract as extract_data
from src.data import transform as transform_data
from src.data import load as load_data


class TabularDataset(Dataset):
    """
    A dataset consisting in tabular data. The data read come from files
    encoding the data as tables.

    Attributes:
        action_settings (Dict): the parameters that define each action from 
        the ETL process.
        data (Dict): the data read from files.
    """

    def __init__(self, action_settings) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters for this specific dataset.

        Args:
            action_settings (Dict): the parameters that define each action 
            from the ETL process.
        """
        super().__init__(action_settings)
        self.data = {}

    def extract(self):
        """
        ETL data extract. Reads data from a file that encodes the data as
        tables (e.g. excel, csv).
        """
        self.data = extract_data.read_data(self.action_settings['extract'])

    def transform(self):
        """
        ETL data transform. This function must be implemented by child classes.
        """
        self.data = transform_data.process_data(
            self.data, self.action_settings['transform'])

    def load(self):
        """
        ETL data load. This function must be implemented by child classes.
        """
        load_data.save_data(self.data, self.action_settings['load'])
