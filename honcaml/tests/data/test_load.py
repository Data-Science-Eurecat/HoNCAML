import os
import shutil
import tempfile
import unittest

import pandas as pd
from honcaml.data import load
from honcaml.exceptions import data as data_exceptions
from honcaml.tests import utils


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_dataframe(self):
        dataset = utils.mock_up_read_dataframe()
        # Extension .csv
        filepath = os.path.join(self.test_dir, 'dataset.csv')
        settings = {'filepath': filepath}
        load.save_dataframe(dataset, settings)
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(os.stat(filepath).st_size > 0)

        # Extension .xlsx
        filepath = os.path.join(self.test_dir, 'dataset.xlsx')
        settings = {'filepath': filepath}
        load.save_dataframe(dataset, settings)
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(os.stat(filepath).st_size > 0)

        # Invalid extension
        filepath = os.path.join(self.test_dir, 'dataset.aaa')
        settings = {'filepath': filepath}
        with self.assertRaises(data_exceptions.FileExtensionException):
            load.save_dataframe(dataset, settings)

    def test_save_model(self):
        settings = {'path': self.test_dir, 'filename': 'model.sav'}
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        model = utils.mock_up_read_model(
            'sklearn', problem_type, model_config)._estimator
        load.save_model(model, settings)
        filepath = os.path.join(settings['path'], settings['filename'])
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(os.stat(filepath).st_size > 0)

    def test_save_predictions(self):
        settings = {'path': self.test_dir}
        df_predictions = pd.DataFrame(
            columns=['f1', 'f2', 'target'],
            data=[[0, 'Str1', 2],
                  [1, 'Str2', 5]])
        load.save_predictions(df_predictions, settings)
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('predictions')
                        for f in files_in_test_dir))
