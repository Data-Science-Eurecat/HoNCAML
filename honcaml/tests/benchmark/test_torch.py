import ray
import unittest

from honcaml.benchmark import torch
from honcaml.exceptions import benchmark as exceptions


class TorchBenchmarkTest(unittest.TestCase):

    def test_clean_search_space_correct(self):
        search_space = {
            'epochs': {
                'method': 'randint',
                'values': [2, 5]
            },
            'layers': {
                'number_blocks': [3, 5],
                'types': ['Linear + ReLU', 'Dropout']
            },
            'loader': {
                'batch_size': {
                    'method': 'randint',
                    'values': [20, 40]
                },
                'shuffle': {
                    'method': 'choice',
                    'values': [True, False]
                }
            },
            'loss': {
                'method': 'choice',
                'values': ['nn.torch.MLELoss', 'nn.torch.L1Loss']
            },
            'optimizer': {
                'module': 'torch.optim.SGD',
                'params': {
                    'lr': 0.001,
                    'momentum': 0.9
                }
            }
        }
        special_keys = ['layers']
        expected = {
            'epochs': ray.tune.search.sample.Integer,
            'layers': {
                'block_1': 'Linear + ReLU',
                'block_2': ray.tune.search.sample.Categorical,
                'block_3': ray.tune.search.sample.Categorical,
                'block_4': ray.tune.search.sample.Categorical,
                'block_5': 'Linear'
            },
            'loader': {
                'batch_size': ray.tune.search.sample.Integer,
                'shuffle': ray.tune.search.sample.Categorical
            },
            'loss': ray.tune.search.sample.Categorical,
            'optimizer': {
                'module': 'torch.optim.SGD',
                'params': {
                    'lr': 0.001,
                    'momentum': 0.9
                }
            }
        }
        result = torch.TorchBenchmark._clean_search_space(
            search_space, special_keys)
        # Assert main structure
        self.assertEqual(list(result), list(expected))

        # Define function to assert sampling types only when necessary
        def assert_dict_types(result, expected):
            for key in result:
                if key in ['module', 'params'] or isinstance(
                        result[key], str):
                    pass
                elif isinstance(result[key], dict):
                    assert_dict_types(result[key], expected[key])
                else:
                    self.assertIsInstance(result[key], expected[key])
        # Run function
        assert_dict_types(result, expected)

    def test_clean_search_space_incorrect_conf(self):
        search_space = {
            'epochs': {
                'test': 'incorrect'
            }
        }
        special_types = ['layers']
        with self.assertRaises(exceptions.IncorrectParameterConfiguration):
            torch.TorchBenchmark._clean_search_space(
                search_space, special_types)

    def test_clean_search_space_layers_different(self):
        number_blocks = [3, 5]
        types = ['Linear + ReLU', 'Dropout']
        expected = {
            'block_1': 'Linear + ReLU',
            'block_2': {
                'method': 'choice',
                'values': ['Linear + ReLU', 'Dropout']
            },
            'block_3': {
                'method': 'choice',
                'values': ['Linear + ReLU', 'Dropout', None]
            },
            'block_4': {
                'method': 'choice',
                'values': ['Linear + ReLU', 'Dropout', None]
            },
            'block_5': 'Linear'
        }
        result = torch.TorchBenchmark._clean_search_space_layers(
            number_blocks, types)
        self.assertEqual(expected, result)

    def test_clean_search_space_layers_equal(self):
        number_blocks = [3, 3]
        types = ['Linear + ReLU', 'Dropout']
        expected = {
            'block_1': 'Linear + ReLU',
            'block_2': {
                'method': 'choice',
                'values': ['Linear + ReLU', 'Dropout']
            },
            'block_3': 'Linear'
        }
        result = torch.TorchBenchmark._clean_search_space_layers(
            number_blocks, types)
        self.assertEqual(expected, result)

    def test_clean_search_space_layers_incorrect_min_blocks(self):
        number_blocks = [1, 7]
        types = ['Linear + ReLU', 'Dropout']
        with self.assertRaises(exceptions.IncorrectNumberOfBlocks):
            torch.TorchBenchmark._clean_search_space_layers(
                number_blocks, types)

    def test_clean_search_space_layers_incorrect_max_blocks(self):
        number_blocks = [5, 3]
        types = ['Linear + ReLU', 'Dropout']
        with self.assertRaises(exceptions.IncorrectNumberOfBlocks):
            torch.TorchBenchmark._clean_search_space_layers(
                number_blocks, types)

    def test_clean_search_space_layers_incorrect_type(self):
        number_blocks = [4, 6]
        types = ['Incorrect', 'Dropout']
        with self.assertRaises(exceptions.TorchLayerTypeDoesNotExist):
            torch.TorchBenchmark._clean_search_space_layers(
                number_blocks, types)

    def test_check_layer_type_correct(self):
        layer_type = 'Linear + ReLU'
        torch.TorchBenchmark._check_layer_type(layer_type)

    def test_check_layer_type_incorrect(self):
        layer_type = 'Incorrect'
        with self.assertRaises(exceptions.TorchLayerTypeDoesNotExist):
            torch.TorchBenchmark._check_layer_type(layer_type)
