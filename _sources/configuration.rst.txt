.. _configuration:

===============
 Configuration
===============

The main difference between the use case of regular users compared to advanced
users is reflected in the configuration of the pipeline.

Conceptually, a pipeline is composed from a modular composition of steps. The
ones available are the following:

* :ref:`data-step`: Responsible for raw data management
* :ref:`model-step`: Responsible for model management
* :ref:`benchmark-step`: Responsible for searching the best model configuration

The configuration should be done through a `YAML <https://yaml.org/spec/>`_
file, which contains all the pipeline and steps options that are considered.

Global
======

First of all, it is necessary to specify global options, which are the
following:

* problem_type: The kind of problem that determines the model/s to use. Valid
  values are: ``classification``, ``regression``.

.. code:: yaml

   global:
       problem_type:

Afterwards, the steps configuration is provided, which is detailed below.

.. _data-step:

Data
====

This step is made up of the following ETL phases:

1. **extract**: Read the raw data.
2. **transform**: Apply a set of data transformations, e.g. normalization, rename
   columns, etc.
3. **load**: Save the processed dataset.

Example of data step in a pipeline file:

.. code:: yaml

   data:
       extract:
         filepath: data/raw/dataset.csv
         features:
           - bedrooms
           - bathrooms
           - sqft_living
         target: price

       transform:
         normalize:
           features:
             module: sklearn.preprocessing.StandardScaler
           target:
             module: sklearn.preprocessing.StandardScaler

       load:
           filepath: data/processed/dataset.

Extract
-------

In the extract phase the possible configurations are the following:

- **filepath** (str, optional): File path of dataset to use. It is an
  optional param. The default value is ``data/raw/dataset.csv``.
- **target** (str): Column name as a target.
- **features** (list, optional): Set of columns to process. It is an optional
  param. When *features* is not indicated, the process uses all columns.

Examples:
^^^^^^^^^

In the following example, the framework reads dataset from
``data/raw/boston_dataset.csv``. Also, it gets ``price`` column as a target.
In addition, it uses ``bedrooms``, ``bathrooms``, ``sqft_living`` as features.

.. code:: yaml

   extract:
     filepath: data/raw/boston_dataset.csv
     features:
       - bedrooms
       - bathrooms
       - sqft_living
     target: price


Transform
---------

In this phase the possible transformations are the following:

Encoding
^^^^^^^^

The parameter **encoding** (dict, optional) defines the dataset encoding for
categorical features through *One Hot Encoding* (OHE) method. This is defined
through two additional parameters within the **encoding** dictionary, which are
**OHE** (boolean), which determines whether to apply the encoding to features
or not, and **features** (list/str, optional), that specify the feature or
features to apply OHE to; if empty and **OHE** is True, OHE would be applied to
all categorical features from the dataset, except for the target.

Examples
^^^^^^^^
The simplest configuration is the following, which means a OHE for all
the categorical features:

.. code:: yaml

   transform:

The following example, the framework applies OHE for all categorical features:

.. code:: yaml

   transform:
     encoding:
      OHE: True

The opposite case would be to not apply OHE to any feature:

.. code:: yaml

   transform:
     encoding:
      OHE: False

Another example is then OHE should be applied only to specific features. In the
following example, OHE is applied only to columns *column_a* and *column_b*.

.. code:: yaml

   transform:
     encoding:
      OHE: True
      features:
        column_a
        column_b


Normalize
^^^^^^^^^

The parameter **normalize** (dict, optional) defines the dataset
normalization. It is possible to normalize nothing, features, target or
both. With **features** parameter, it defines which normalization apply to
features. Furthermore, with **target** parameter, it defines the target
normalization. If the transform step contains an empty **normalize** key, it
uses a ``sklearn.preprocessing.StandardScaler`` for features and target as
default. On the other hand, if **normalize** key does not exist, no
normalization is applied. If only features or target (but not both) are to be
normalized, empty settings should be provided for the part that does not
require normalization.

-  **target** (list, optional): Column name as a target. It is an
   optional param. The default value is ``target``.
-  **features** (list, optional): Set of columns to process. It is an
   optional param. When empty, the process uses all columns.

For any of the previous mentioned, there are three children keys:

- **module** (str, optional): Normalization module to apply. Right now,
  ``sklean.preprocessing.StandardScaler`` is the only one supported.
- **params** (dict, optional): Specific parameters of the previous
  module. Should be specified as key-value pairs.
- **columns** (list, optional): Columns to be considered for normalization. By
  default, all of features/target if empty, depending on the context.


Examples
^^^^^^^^

The simplest configuration is the following, which means no normalization:

.. code:: yaml

   transform:

In the example below, the framework applies a default normalization parameters
(``sklearn.preprocessing.StandardScaler`` for both target and features).

.. code:: yaml

   transform:
     normalize:

If only features or target are to be normalized, just an empty module should be
provided for target.

.. code:: yaml

   transform:
     normalize:
       target:
         module:

In the example below, the framework uses a
``sklearn.preprocessing.StandardScaler`` for normalize target and
features. In the case of features, normalization is applied considering std,
and only for columns named *column_a* and *column_b*.

.. code:: yaml

   transform:
     normalize:
       features:
         module: sklearn.preprocessing.StandardScaler
         params:
           with_std: True
           columns:
             - column_a
             - column_b

Load
----

In load phase the possible configurations are the following:

- **filepath** (str, optional): file path to store processed dataset.

Examples
^^^^^^^^

The simplest configuration is the following:

.. code:: yaml

   load:

When **load** phase is empty, the framework does not save the processed
dataset.

The following example, the framework stores the processed data in
``data/processed/dataset.csv``.

.. code:: yaml

   load:
     filepath: data/processed/dataset.csv

.. _model-step:

Model
=====

This step is responsible for model management.

It is made up for the following ETL phases:

- **extract**: The purpose of this phase is to read a previously saved model.
- **transform**: This phase applies the common model functions:
  training, testing and cross-validation.
- **load**: Saves the initialized model.

Extract
-------

In extract phase the possible configurations are the following:

- **filepath** (str [#comp1]_): File path of model to read.

.. [#comp1]

   Compulsory for predict pipelines, and excluded in the rest of pipeline types.

Transform
---------

This phase applies the common model functions: fit, predict and
cross-validation. The available configurations are the following:

- **fit** (dict [#comp2]_): Requests a model training on the current dataset. It
  may have the following additional information:

  - **estimator** (dict, optional): Sppecifies the estimator and its
    hyperparameters. Consists of the following:

    - **module** (str, optional): Learner module to use.
    - **params** (dict, optional): Additional parameters to pass to module
      class.

    Available models are the ones `available from sklearn
    <https://scikit-learn.org/stable/supervised_learning.html>`_, and of
    course just the ones related to the problem type specified.
    Default models are ``sklearn.ensemble.RandomForestRegressor`` for
    regression and ``sklearn.ensemble.RandomForestClassifier`` for
    classification problems, both with ``n_estimator`` equal to 100.

  - **cross_validation** (dict, optional): Defines which cross-validation
    strategy to use for training the model. Dictionary may have the following
    keys:

    - **module** (str, optional): Cross validation module to use.
    - **params** (dict, optional): Additional parameters to pass to module
      class.

    Any cross validation method in `sklearn cross-validation
    <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`_
    should work, provided that it follows their consistent structure.
    Default: ``sklearn.model_selection.KFold`` with 3 splits.

  - **metrics** (list): a list of metrics to evaluate the model.
    Any metric that exists in `sklearn.metrics
    <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
    is allowed, of course that apply to the problem type; only the function
    name is required.  Default values are ``mean_squared_error``,
    ``mean_absolute_percentage_error``, ``median_absolute_error``,
    ``mean_absolute_error``, ``root_mean_squared_error`` for regression
    problems and ``accuracy_score``, ``precision_score``, ``recall_score``,
    ``specificity_score``, ``f1_score`` and ``roc_auc_score`` for
    classification problems.

    It is even possible to define custom metrics. For this, what is needed just
    to define a function named ``compute_{metric_name}_metric`` in the file
    ``honcaml/models/evaluate.py``, being {metric_name} the name of the
    metric, and having as input parameters the series of true values, and the
    series of predicted ones, in this order (there are already a couple of
    examples). Then, it is just a matter of include the metric name in the
    configuration.

    Both options have the possibility to pass additional parameters to the
    metric function, by specifying the metric as a dictionary instead of a
    single string. The dictionary key would be the metric name, whereas its
    values would refer to function parameters.

- **predict** (dict [#comp3]_): Requests to run predictions over the dataset.

  - **path** (str, optional): Directory where the predictions will be
    stored. Default value: ``data/processed``.

.. [#comp2]

   Compulsory for fit pipelines, and excluded for predict pipelines. Related to
   benchmark pipelines, see the details in :ref:`benchmark-step`.

.. [#comp3]

   Compulsory for predict pipelines, and excluded for the rest of pipeline
   types.

Examples
^^^^^^^^

The following snippet shows an example of an advanced model transform
definition:

.. code:: yaml

    transform:
      fit:
        estimator:
          module: sklearn.ensemble.RandomForestRegressor
          params:
            n_estimators: 100
        cross_validation:
          module: sklearn.model_selection.KFold
          params:
            n_splits: 2

.. _deep-learning-models:

Deep learning models
^^^^^^^^^^^^^^^^^^^^

Deep learning models implemented in torch require a specific format, different
from sklearn based models or similar, in which parameters are passed directly
when instantiating the model class.

First of all, **module** key should have just as value ``torch`` in order to
indicate that a neural net will be used as estimator. Within the **params**
key, the following keys should be specified [#comp4]_:

- **epochs** (int): Number of training epochs.

- **layers** (list) Layers configuration; the structure of each one is:

  - **module** (str): Layer module to use.
  - **params** (dict, optional [#comp5]_): Additional parameters to pass to
    layers.

  In the case of linear layers, as the parameter **in_features** is dependent
  on previous layers, only **out_features** is required; however, if the last
  layer of the neural net is another linear layer, no **out_features** should
  be provided, as dimension will be inferred from targets.

- **loader**: (dict): Specifies data loader options to use. Internal keys:

    - **batch_size** (int): Number of rows to consider for each batch.
    - **shuffle** (bool): Whether to shuffle data at every epoch.

- **loss** (dict): Loss to consider; requires the following:

  - **module** (str): Loss module to use.
  - **params** (dict, optional): Additional parameters to pass to module.

- **optimizer** (dict): Optimizer to consider; requires the following:

  - **module** (str): Optimizer module to use.
  - **params** (dict, optional): Additional parameters to pass to module.

An example of a training configuration for a deep learning model would be:

.. code:: yaml

  model:
    transform:
      fit:
        estimator:
          module: torch
          params:
            epochs: 3
            layers:
              - module: torch.nn.Linear
                params:
                  out_features: 64
              - module: torch.nn.ReLU
              - module: torch.nn.Linear
                params:
                  out_features: 32
              - module: torch.nn.Dropout
              - module: torch.nn.Linear
            loader:
              batch_size: 20
              shuffle: True
            loss:
              module: torch.nn.MSELoss
            optimizer:
              module: torch.optim.SGD
              params:
                lr: 0.001
                momentum: 0.9

.. [#comp4]

   All options are required for training and benchmark pipelines, whereas
   dataloader is the only one required by predict pipelines.

.. [#comp5]

   Optional for all layer types except for linear ones, except for the last
   layer if it is linear.

Load
----

In load phase the possible configurations are the following:

- **filepath** (str, required): Directory and file name where the model will be saved.
  If the user specifies the file name as ``{autogenerate}.sav``, the filename is
  generated by the framework following the following
  convention: ``{model_type}.{execution_id}.sav``
  Otherwise, if the user specifies a custom name, the file is saved with that name.
  The supported formats for saving a model include the extension ``.sav``

- **results** (str, [#comp6]_): Directory where to store training cross
  validation results; generated file will have the following format:
  ``{results}/{execution_id}/results.csv``. If not set, results will not be
  exported.

.. [#comp6]

   Optional for train pipelines, and excluded for the rest of pipeline
   types.

.. _benchmark-step:

Benchmark
=========

This step is responsible for searching the best model configuration.

It is made up for the following ETL phases:

- **transform**: this phase runs an hyperparamater search algorithm for each
  specified model. Furthermore, it gets the best model configuration.

- **load**: it saves the best configuration into a yaml file.

Apart from obtaining the best model configuration, it is possible to train the
best model through appending a model key after the benchmark step, taking
advantage of the modular definition of the solution:

.. code:: yaml

   global:
     problem_type: regression

   steps:
     data:
       extract:
         filepath: {Input data}
         target: {Target}

  benchmark:
    transform:
    load:
      path: {gReports path}

  model:
    transform:
      fit:
    load:
      path: {Path to store best model}

Transform
---------

This phase runs an hyperparameter search algorithm for each model defined in
pipeline file. Furthermore, the user can define a set of metrics to evaluate
the experiments, the model's hyperparamaters to tune, the strategy to split
train and test data and parameters of search algorithm.

The available configurations are the following:

- **models** (dict, optional): Dictionary of models and hyperparameters to
  search for best configuration. Each entry of the list refers to a model to
  benchmark. Keys should be the following:

  - **{model_name}** (dict, optional): Name of model module,
    e.g. ``sklearn.ensemble.RandomForestRegressor``.

  Within each module, there should be as many keys as model parameters to
  search:

    - **{hyperparameter}** (dict, optional): Name of hyperparameter,
      e.g. ``n_estimators``. Within each hyperparameter, the following needs to
      be specified:

      - **method** (str, optional): Method to consider for searching
        hyperparameter values.
      - **values** (tuple/list, optional): Values to consider for hyperparameter
        search, passed to specified method.

  Available methods and value parameters are defined in the `search space
  <https://docs.ray.io/en/latest/tune/api/search_space.html>`_.  The default
  models and hyperparameters for each type of problem are defined at
  *honcaml/config/defaults/search_spaces.py*.

  In case of deep learning models, the name of the model to use is ``torch``,
  and there is a specific chapter to detail the required configuration in
  :ref:`deep-learning-benchmark`.

- **cross_validation** (dict, optional): defines which cross-validation
  strategy to use for training each model. Dictionary may have the following
  keys:

  - **module** (str, optional): Cross validation module to use.
  - **params** (dict, optional): Additional parameters to pass to module class.

  Any cross validation method in `sklearn cross-validation
  <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`_
  should work, provided that it follows their consistent structure.
  Default: ``sklearn.model_selection.KFold`` with 3 splits.

- **metrics** (list/str, optional): a list of metrics to report in the
  benchmark process, or a single one. Actually, reported metrics may be
  appended with the one specified in tuner settings, if the latter is different
  (as it is the one used to select the best model configuration). Any metric
  that exists in `sklearn.metrics
  <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
  is allowed, of course that apply to the problem type; only the function name
  is required.  Default values are ``mean_squared_error``,
  ``mean_absolute_percentage_error``, ``median_absolute_error``,
  ``mean_absolute_error``, ``root_mean_squared_error`` for regression problems
  and ``accuracy_score``, ``precision_score``, ``recall_score``,
  ``specificity_score``, ``f1_score`` and ``roc_auc_score`` for classification
  problems.

  It is even possible to define custom metrics. For this, what is needed just
  to define a function named ``compute_{metric_name}_metric`` in the file
  ``honcaml/models/evaluate.py``, being {metric_name} the name of the
  metric, and having as input parameters the series of true values, and the
  series of predicted ones, in this order (there are already a couple of
  examples). Then, it is just a matter of include the metric name in the
  configuration.

- **tuner** (dict): defines the configuration of tune process. Their options
  are the following:

  - **search_algorithm** (dict, optional): Specifies the algorithm to perform
    the search. Consists of the following:

    - **module** (str, optional): Algorithm module to use.
    - **params** (dict, optional): Additional parameters to pass to module
      class.

  For all available options, see `the search algorithms documentation
  <https://docs.ray.io/en/latest/tune/api/suggestion.html>`_.
  Default is ``ray.tune.search.optuna.OptunaSearch``.

  - **tune_config** (dict, optional): Parameters to pass to tuner config
    object, specified as key-value pairs. For available options, see `TuneConfig
    documentation
    <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html>`_.
  - **run_config** (dict, optional): Parameters to be used during run,
    specified as key-value pairs. For available options, see `RunConfig
    documentation
    <https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html>`_.
  - **scheduler** (dict, optional): Allows to define different strategies
    during the search process. Consists of the following:

    - **module** (str, optional): Algorithm module to use.
    - **params** (dict, optional): Additional parameters to pass to module
      class.

  For all available options, see `schedulers documentation
  <https://docs.ray.io/en/latest/tune/api/schedulers.html>`_.

Examples
^^^^^^^^

The following snippet shows an example of an advanced benchmark transform
definition:

.. code:: yaml

   metrics:
     - mean_squared_error
     - mean_absolute_error
     - root_mean_square_error
   models:
     sklearn.ensemble.RandomForestRegressor:
       n_estimators:
         method: randint
         values: [2, 110]
       max_features:
         method: choice
         values: [sqrt, log2, 1]
     sklearn.linear_model.LinearRegression:
       fit_intercept:
         method: choice
         values: [True, False]
   cross_validation:
     module: sklearn.model_selection.KFold
     params:
       n_splits: 2
   tuner:
     search_algorithm:
       module: ray.tune.search.optuna.OptunaSearch
     tune_config:
       num_samples: 5
       metric: root_mean_squared_error
       mode: min
     run_config:
       stop:
         training_iteration: 2
     scheduler:
       module: ray.tune.schedulers.HyperBandScheduler

.. _deep-learning-benchmark:

Deep learning benchmark
^^^^^^^^^^^^^^^^^^^^^^^

Deep learning models, in a benchmark pipeline, require a specific format, due
to the fact that models require a custom format as well (it is advisable to
review their structure in :ref:`deep-learning-models`). The main structure
should be the same:

- **epochs** (dict): Typical keys **method** (with value ``randint``) and
  **values** should be specified.

- **layers** (dict) Layer structure to benchmark; this key is the only one with
  a completely different structure than specified in deep learning models; this
  is because the approach for benchmarking them is through what are called
  blocks. Blocks are a predefined combination of layers that will be shuffled
  with a specific layer to generate combinations to benchmark. For example, one
  block could be a linear layer + rectified linear unit, and another one could
  be a dropout layer. The required structure is the following:

  - **number_blocks** (list): List of two values, which is the minimum and
    maximum number of blocks considered for the models.
  - **types** (list): List of strings that specify succession of layer types to
    be considered as blocks, assuming that their names are contained within
    `torch nn module <https://pytorch.org/docs/stable/nn.html>`_. Blocks that
    contain a sequence of layers should join their names with the symbol ``+``.
  - **params** (dict, optional): In case some layer types require specific
    parameters to be benchmarked, they should be informed within this key. The
    structure to follow is the following:

    - **{layer name}** (str): Layer name, as specified in **types**.

      - **{parameter name}** (str): Name of parameter to be benchmarked. Its
        internal structure should have the typical benchmark structure,
        **method** and **values**.

- **loader**: (dict): Should still have both keys, **batch_size** and
  **shuffle**, and each of them follow the standard benchmark structure
  (**method** and **values**).

- **loss** (dict): Loss to consider; requires the following:

  - **method** (str): Should be equal to ``choice``.
  - **values** (list): For each possible option to consider, specify the
    following:

    - **module** (str): Loss module.
    - **params** (dict, optional): Parameters to benchmark for the specific
      module, in case there are any. Each of them should have the standard
      structure **method** and **values**.

- **optimizer** (dict): Optimizer to consider; requires the following:

  - **method** (str): Should be equal to ``choice``.
  - **values** (list): For each possible option to consider, specify the
    following:

    - **module** (str): Optimizer module.
    - **params** (dict, optional): Parameters to benchmark for the specific
      module, in case there are any. Each of them should have the standard
      structure **method** and **values**.

An example of a benchmark configuration for deep learning models would be:

.. code:: yaml

  benchmark:
    transform:
      models:
        torch:
          epochs:
            method: randint
            values: [2, 5]
          layers:
            number_blocks: [3, 6]
            types:
              - Linear + ReLU
              - Dropout
            params:
              Dropout:
                p:
                  method: uniform
                  values: [0.4, 0.6]
          loader:
            batch_size:
              method: randint
              values: [20, 40]
            shuffle:
              method: choice
              values:
                - True
                - False
          loss:
            method: choice
            values:
              - module: torch.nn.MSELoss
              - module: torch.nn.L1Loss
          optimizer:
            method: choice
            values:
              - module: torch.optim.SGD
                params:
                  lr:
                    method: loguniform
                    values: [0.001, 0.01]
                  momentum:
                    method: uniform
                    values: [0.5, 1]
              - module: torch.optim.Adam
                params:
                  lr:
                    method: loguniform
                    values: [0.001, 0.1]
                  eps:
                    method: loguniform
                    values: [0.0000001, 0.00001]

Load
----

In load phase the possible configurations are the following:

- **path** (str): Folder in which to store benchmark results.
- **save_best_config_params** (bool, optional): Whether to store a yaml file
  with best model configuration or not, within specified **path**.
