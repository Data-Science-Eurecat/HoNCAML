import torch

from honcaml.benchmark import base
from honcaml.exceptions import benchmark as exceptions
from honcaml.tools.startup import logger


class TorchBenchmark(base.BaseBenchmark):

    @classmethod
    def _clean_search_space(cls, search_space: dict,
                            special_keys: list = ['layers']) -> dict:
        """
        Given a dict with a search space for a model, this function gets the
        module of model to import and the hyperparameters search space and
        ensures that method exists.

        Args:
            search_space (Dict): a dict with the search space to explore
            special_keys (list): Keys to treat in a specific way

        Returns:
        (Dict): a dict where for each hyperparameter the corresponding
        method to generate all possible values during the search.
        """
        cleaned_search_space = {}
        for hyper_parameter, space in search_space.items():
            logger.debug(f'Cleaning hyperparameter {hyper_parameter}')

            # Standard values of space should have method/value keys
            if 'method' not in space:
                # Special keys are handled differently
                if hyper_parameter in special_keys:
                    func_obj = getattr(
                        cls, f"_clean_search_space_{hyper_parameter}")
                    # Several parameters are introduced at once
                    new_parameters = func_obj(**space)
                    cleaned_search_space[
                        hyper_parameter] = cls._clean_search_space(
                            new_parameters, special_keys)
                # Specific parameter set if module/params combination is used
                elif ('module' and 'params' in space) or (
                        'block' in hyper_parameter):
                    cleaned_search_space[hyper_parameter] = space
                # Recursive call in case there are nested options
                elif isinstance(space, dict):
                    cleaned_search_space[
                        hyper_parameter] = cls._clean_search_space(space)
                else:
                    raise exceptions.IncorrectParameterConfiguration(
                        hyper_parameter)
            else:
                cleaned_search_space[
                    hyper_parameter] = super()._clean_parameter_search_space(
                        space)

        return cleaned_search_space

    @classmethod
    def _clean_search_space_layers(
            cls, number_blocks: list, types: list[str],
            first_block: str = 'Linear + ReLU',
            last_block: str = 'Linear') -> dict:
        """
        Clean search space related to neural net layers.

        The core idea is that several layer blocks will be optimized, and they
        have a minimum and maximum number. Therefore, in order to be consistent
        with the optimizer, the output should consider as much parameters as
        maximum number of blocks excluding first and last, and each of them can
        be of the defined types.

        Note: It is important to note that, as first and last layer should have
        a specific structure, they are hard-coded within the method.

        Args:
            number_blocks: Minimum and maximum number of blocks considered
            types: Possible types of blocks. Should have a specific format, in
                which all layers should be strings with name of real layer
                types from `torch.nn` module:
                 - Simple layer -> {layer}
                 - Sequential layers -> {layer1 + ... + layern}
            first_block: Type of first block
            last_block: Type of last block

        Returns:
            Search space correctly formatted for the optimizer
        """
        # Ensure number of blocks is enough
        if (number_blocks[0] < 2) or (
                number_blocks[0] > number_blocks[1]):
            raise exceptions.IncorrectNumberOfBlocks(number_blocks)
        # Ensure types are correct
        for layer_type in types:
            cls._check_layer_type(layer_type)
        parameters = {}
        parameters['block_1'] = first_block
        initial_layer = 2
        layer_types = types
        for i in range(number_blocks[1] - 2):
            layer_num = initial_layer + i
            if layer_num > number_blocks[0] - 1:
                layer_types = types + [None]
            parameters[f'block_{layer_num}'] = {
                'method': 'choice', 'values': layer_types}
        layer_num += 1
        parameters[f'block_{layer_num}'] = last_block
        return parameters

    @staticmethod
    def _check_layer_type(layer_type: str) -> None:
        """
        Check if layer type is correct considering specifications.

        Args:
            layer_type: Type of layer

        Raises:
            Exception TorchLayerTypeDoesNotExist if layer is incorrect
        """
        parts_to_check = layer_type.replace(' ', '').split('+')
        for part in parts_to_check:
            try:
                getattr(torch.nn, part)
            except AttributeError:
                raise exceptions.TorchLayerTypeDoesNotExist(part)
