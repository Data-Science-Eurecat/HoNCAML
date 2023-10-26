import ray
import unittest

from honcaml.benchmark import torch
from honcaml.exceptions import benchmark as exceptions


class TorchBenchmarkTest(unittest.TestCase):

    def test_clean_search_space_correct(self):
        search_module = ray.tune.search.sample
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
                'method': 'choice',
                'values': [
                    {
                        'module': 'torch.optim.SGD',
                        'params': {
                            'lr': {
                                'method': 'loguniform',
                                'values': [0.001, 0.01]
                            },
                            'momentum': {
                                'method': 'uniform',
                                'values': [0.5, 1]
                            }
                        }
                    },
                    {
                        'module': 'torch.optim.Adam',
                        'params': {
                            'lr': {
                                'method': 'loguniform',
                                'values': [0.001, 0.1]
                            },
                            'eps': {
                                'method': 'loguniform',
                                'values': [0.0000001, 0.00001]
                            }
                        }
                    }
                ]
            }
        }
        special_keys = ['layers']
        expected = {
            'epochs': ray.tune.search.sample.Integer,
            'layers': {
                'blocks': {
                    'block_1': 'Linear + ReLU',
                    'block_2': search_module.Categorical,
                    'block_3': search_module.Categorical,
                    'block_4': search_module.Categorical,
                    'block_5': 'Linear'
                }
            },
            'loader': {
                'batch_size': search_module.Integer,
                'shuffle': search_module.Categorical
            },
            'loss': search_module.Categorical,
            '[optimizer]-[torch.optim.SGD]-lr': search_module.Float,
            '[optimizer]-[torch.optim.SGD]-momentum': search_module.Float,
            '[optimizer]-[torch.optim.Adam]-lr': search_module.Float,
            '[optimizer]-[torch.optim.Adam]-eps': search_module.Float,
            'optimizer': search_module.Categorical,
        }
        result = torch.TorchBenchmark.clean_search_space(
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
                    print('key', key)
                    print('result', result)
                    print('expected', expected)
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
            torch.TorchBenchmark.clean_search_space(
                search_space, special_types)

    def test_clean_search_space_layers_different(self):
        number_blocks = [3, 5]
        types = ['Linear + ReLU', 'Dropout']
        params = {'Dropout': {'method': 'uniform', 'values': [0.4, 0.6]}}
        expected = {
            'blocks': {
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
            },
            'params': {'Dropout': {'method': 'uniform', 'values': [0.4, 0.6]}}
        }
        result = torch.TorchBenchmark._clean_search_space_layers(
            number_blocks, types, params)
        self.assertEqual(expected, result)

    def test_clean_search_space_layers_equal(self):
        number_blocks = [3, 3]
        types = ['Linear + ReLU', 'Dropout']
        expected = {
            'blocks': {
                'block_1': 'Linear + ReLU',
                'block_2': {
                    'method': 'choice',
                    'values': ['Linear + ReLU', 'Dropout']
                },
                'block_3': 'Linear'
            }
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

    def test_invalidate_experiment_false(self):
        search_space = {
            'layers': {
                'blocks': {
                    'block_1': 'Linear + ReLU',
                    'block_2': 'Linear + ReLU',
                    'block_3': 'Dropout',
                    'block_4': 'Linear'
                }
            }
        }
        expected = False
        result = torch.TorchBenchmark.invalidate_experiment(search_space)
        self.assertEqual(expected, result)

    def test_invalidate_experiment_true(self):
        search_space = {
            'layers': {
                'blocks': {
                    'block_1': 'Linear + ReLU',
                    'block_2': 'Dropout',
                    'block_3': 'Dropout',
                    'block_4': 'Linear'
                }
            }
        }
        expected = True
        result = torch.TorchBenchmark.invalidate_experiment(search_space)
        self.assertEqual(expected, result)
