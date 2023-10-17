import numpy as np
from sklearn import compose, pipeline
import torch
from typing import Callable, Dict, List, Tuple

from honcaml.data import extract, load, normalization
from honcaml.models import base, evaluate
from honcaml.tools import custom_typing as ct
from honcaml.tools import utils
from honcaml.tools.startup import logger


class TorchModel(base.BaseModel):
    """
    Torch model wrapper.
    """

    def __init__(self, problem_type: str) -> None:
        """
        Class constructor which initializes the base class.

        Args:
            problem_type (str): The kind of problem to be addressed. Valid
                values are `regression` and `classification`.
        """
        super().__init__(problem_type)
        self._estimator = None
        self._pipeline = None

    @property
    def estimator(self) -> ct.SklearnModelTyping:
        """
        Getter method for the '_estimator' attribute.

        Returns:
            '_estimator' current value.
        """
        return self._estimator

    @property
    def estimator_type(self) -> str:
        """
        Getter method for the '_estimator_type' attribute.

        Returns:
            '_estimator_type' current value.
        """
        return self._estimator_type

    def read(self, settings: Dict) -> None:
        """
        Read an estimator from disk.

        Args:
            settings: Parameter settings defining the read operation.
        """
        self._estimator = extract.read_model(settings)

    @staticmethod
    def _import_estimator(
            layers_config: dict, whole_input_dim: int,
            whole_output_dim: int) -> Callable:
        """
        Given a dict with model configuration, this function import the model
        and, it creates a new instance with the hyperparameters.

        Args:
            layers_config (dict): a dict with layers information
            whole_input_dim (int): Input dimension of the whole network
            whole_output_dim (int): Output dimension for the whole network

        Returns:
            (Callable): an instance of model with specific hyperparameters.
        """
        layers_ops = []
        prev_out_features = None
        for i, layer in enumerate(layers_config):
            if 'params' not in layer:
                layer['params'] = {}
            layer_type = layer['module'].split('.')[-1]
            if layer_type == 'Linear':
                # If first layer
                if i == 0:
                    layer['params']['in_features'] = whole_input_dim
                # If last layer
                elif i == len(layers_config) - 1:
                    layer['params']['in_features'] = prev_out_features
                    layer['params']['out_features'] = whole_output_dim
                else:
                    layer['params']['in_features'] = prev_out_features
                prev_out_features = layer['params']['out_features']
            layer_op = utils.import_library(**layer)
            layers_ops.append(layer_op)
        model = torch.nn.Sequential(*layers_ops).to(torch.device('cpu'))
        return model

    def build_model(self, model_config: Dict,
                    normalizations: normalization.Normalization,
                    features: List, target: List) -> None:
        """
        Creates the torch estimator. It builds a sklearn pipeline to handle
        preprocessing, plus the torch model separately.

        Args:
            model_config: Model configuration, i.e. module and its
                hyperparameters.
            normalizations: Definition of normalizations that applies to
                the dataset during the model pipeline.
            features: List of features for model.
            target: List of targets for model.
            **kwargs: Extra parameters.
        """
        pipeline_steps = []
        pre_process_transformations = []
        # Features preprocessing
        if normalizations is not None and normalizations.features:
            features_norm = ('features_normalization',
                             normalizations.features_normalizer,
                             normalizations.features)
            pre_process_transformations.append(features_norm)
        if pre_process_transformations:
            pre_process = compose.ColumnTransformer(
                transformers=pre_process_transformations,
                remainder='passthrough')
            pipeline_steps.append(('pre_process', pre_process))
        self._pipeline = pipeline.Pipeline(pipeline_steps)
        # Target preprocessing
        if normalizations is not None and normalizations.target:
            target_norm = ('target_normalization',
                           normalizations.target_normalizer,
                           normalizations.target)
            target_pre_process = compose.ColumnTransformer(
                transformers=[target_norm])
            self._target_pipeline = pipeline.Pipeline(target_pre_process)
        # Model
        input_dim, output_dim = self._retrieve_input_and_output_dims(
            features, target)
        self._estimator = self._import_estimator(
            model_config['params']['layers'], input_dim, output_dim)
        logger.debug(f'Model object {self._estimator}')

    @staticmethod
    def _retrieve_input_and_output_dims(
            features: List, target: List) -> Tuple[int]:
        """
        Retrieve input and output dimensions of model from dataset shape.

        Args:
            dataset: Tranining dataset
            target: Target columns

        Returns:
            - Model input dimension
            - Model output dimension
        """
        output_dim = len(target)
        input_dim = len(features)
        return (input_dim, output_dim)

    def fit(self, x: ct.Dataset, y: ct.Dataset, loader: Dict, loss: str,
            optimizer: Dict, epochs: int, **kwargs: Dict) -> None:
        """
        Trains the estimator on the specified dataset. Must be implemented by
        child classes.

        Args:
            x: Dataset features.
            y: Dataset target.
            loader: Options for dataloader.
            loss: Name of loss function to use.
            optimizer: Optimizer module and params to use.
            epochs: Number of epochs for training.
            **kwargs: Extra parameters.
        """
        dataset = TorchTrainDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, **loader)
        criterion = utils.import_library(**loss)
        # Define dictionary to pass to optimizer
        # Estimator parameters should be included
        optimizer = utils.import_library(
            **optimizer, mand_argument=self._estimator.parameters())

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward + backward + optimize
                outputs = self._estimator(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                logger.debug(
                    f'[{epoch + 1}, {i + 1}] loss: {running_loss}')

    def predict(
            self, x: ct.Dataset, loader: Dict, **kwargs: Dict) -> List:
        """
        Uses the estimator to make predictions on the given dataset features.
        Must be implemented by child classes.

        Args:
            x: Dataset features.
            loader: Options for dataloader.
            **kwargs: Extra parameters.

        Returns:
            Resulting predictions from the estimator.
        """
        dataset = TorchTestDataset(x)
        loader = torch.utils.data.DataLoader(dataset, **loader)
        predictions = torch.tensor([])
        with torch.no_grad():
            for data in loader:
                outputs = self._estimator(data)
                # In case of classification?
                if self.estimator_type == 'classifier':
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    predicted = outputs.data
                predictions = torch.cat((predictions, predicted), 0)
        predictions = np.array(predictions)
        return predictions

    @staticmethod
    def _append_predictions(
            predictions: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Append batch predictions to global ones, considering multiple targets
        if necessary.

        Args:
            predictions: Global predictions until previous batch.
            outputs: Batch output.

        Returns:
            Updated predictions.
        """
        _, predicted = torch.max(outputs.data, 1)
        predictions = torch.cat((predictions, predicted), 0)
        return predictions

    def evaluate(self, x: ct.Dataset, y: ct.Dataset, metrics: List,
                 loader: Dict, **kwargs: Dict) -> Dict:
        """
        Evaluates the estimator on the given dataset.

        Args:
            x: Dataset features.
            y: Dataset target.
            metrics: Metrics to be computed.
            loader: Options for dataloader.
            **kwargs: Extra parameters.

        Returns:
            Resulting metrics from the evaluation.
        """
        y_pred = self.predict(x, loader)
        metrics = utils.ensure_input_list(metrics)
        metrics = evaluate.compute_metrics(y, y_pred, metrics)
        return metrics

    def save(self, settings: Dict) -> None:
        """
        Stores the estimator to disk.

        Args:
            settings: Parameter settings defining the store operation.
        """
        settings['filename'] = utils.generate_unique_id(
            base.ModelType.torch) + '.sav'
        load.save_model(self._estimator, settings)


class TorchTrainDataset(torch.utils.data.Dataset):
    """
    Dataset class needed for TorchModel to be trained.
    """
    def __init__(self, x: ct.Dataset, y: ct.Dataset):
        self.x = torch.tensor(np.array(x), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        x_val = self.x[idx]
        y_val = self.y[idx]
        return x_val, y_val

    def __len__(self):
        return len(self.y)


class TorchTestDataset(torch.utils.data.Dataset):
    """
    Dataset class needed for TorchModel to predict.
    """
    def __init__(self, x: ct.Dataset):
        self.x = torch.tensor(np.array(x), dtype=torch.float32)

    def __getitem__(self, idx):
        x_val = self.x[idx]
        return x_val

    def __len__(self):
        return len(self.x)
