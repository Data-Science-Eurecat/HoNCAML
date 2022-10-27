.. _configuration:

===============
 Configuration
===============

The main difference between the use case of regular users compared to advanced
users is reflected in the configuration of the pipeline.

Conceptually, a pipeline is composed from a modular composition of steps. The
ones available are the following:

* :ref:`data-step`: Responsible for raw data management
* :ref:`benchmark-step`: Responsible for searching the best model configuration
* :ref:`model-step`: Responsible for model management

The configuration should be done through a `YAML <https://yaml.org/spec/>`_
file, which contains all the pipeline and steps options that are considered.

Global
======

First of all, it is necessary to specify global options, which are the
following:

* problem_type: The kind of problem that determines the model/s to use. Valid
  values are: ``classification``, ``regression``.
* metrics_folder: Folder in which all execution results will be stored,
  which will contain the id of their execution in order to differentiate them.

.. code:: yaml

   global:
       problem_type:
       metrics_folder:

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
         target:
           - price
         features:
           - bedrooms
           - bathrooms
           - sqft_living 

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

- **filepath** (str, optional): file path of dataset to use. It is an
  optional param. The default value is ``data/raw/dataset.csv``.
- **target** (list, optional): column name as a target. It is an optional
  param. The default value is ``target``.
- **features** (list, optional): a set of columns to process. It is an optional
  param. When *features* is not indicated, the process uses all columns.

Examples:
^^^^^^^^^

The simplest configuration is the following:

.. code:: yaml

   extract:
     filepath: data/raw/boston_dataset.csv
     target:
       - price

In the following example, the framework reads dataset from
``data/raw/boston_dataset.csv``. Also, it gets ``price`` column as a target.
In addition, it uses ``bedrooms``, ``bathrooms``, ``sqft_living`` as features.

.. code:: yaml

   extract:
     filepath: data/raw/boston_dataset.csv
     target:
       - price
     features:
       - bedrooms
       - bathrooms
       - sqft_living

Transform
---------

In this phase the possible transformations are the following:

Normalize
^^^^^^^^^

The parameter **normalize** (dict, optional) defines the dataset
normalization. It is possible to normalize nothing, features, target or
both. With **features** parameter, it defines which normalization apply to
a features. Furthermore, with **target** parameter, it defines the target
normalization. If the transform step contains an empty **normalize** key,
it uses a ``sklearn.preprocessing.StandardScaler`` for features and target
as default. On the other hand, if **normalize** key does not exist, no
normalization is applied.

-  **target** (list, optional): column name as a target. It is an
   optional param. The default value is ``target``.
-  **features** (list, optional): a set of columns to process. It is an
   optional param. When empty, the process uses all columns.

Examples
^^^^^^^^

The simplest configuration is the following:

.. code:: yaml

   transform:

When **transform** phase is empty, it does not apply any transformation.

In the example below, the framework applies a default normalization
parameters.

.. code:: yaml

   transform:
     normalize:

In the example below, the framework uses a
``sklearn.preprocessing.StandardScaler`` for normalize only target.

.. code:: yaml

   transform:
     normalize:
       target:
         module: sklearn.preprocessing.StandardScaler

The following example, the framework uses a
``sklearn.preprocessing.StandardScaler`` for normalize only features.

.. code:: yaml

   transform:
     normalize:
       features:
         module: sklearn.preprocessing.StandardScaler

In the example below, the framework uses a
``sklearn.preprocessing.StandardScaler`` for normalize target and
features.

.. code:: yaml

   transform:
     normalize:
       features:
         module: sklearn.preprocessing.StandardScaler
       target:
         module: sklearn.preprocessing.StandardScaler

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

.. _benchmark-step:

Benchmark
=========

This step is responsible for searching the best model configuration.

It is made up for the following ETL phases:

- **transform**: this phase runs an hyperparamater search algorithm for each
  specified model. Furthermore, it gets the best model configuration.
- **load**: it saves the best configuration into a yaml file.

The following example shows all keys that can be specified in a pipeline
file:

.. code:: yaml

    benchmark:
        transform:
          metrics:
            - mean_squared_error
            - mean_absolute_percentage_error
            - median_absolute_error
            - r2_score
            - mean_absolute_error
            - root_mean_square_error

          models:
            - module: sklearn.ensemble.RandomForestRegressor
              search_space:
                n_estimators:
                  method: randint
                  values: [ 2, 110 ]
                max_features:
                  method: choice
                  values:
                    - sqrt
                    - log2
                    - 1
            - module: sklearn.linear_model.LinearRegression
              search_space:
                fit_intercept:
                  method: choice
                  values:
                    - True
                    - False

          cross_validation:
            strategy: k_fold
            n_splits: 2
            shuffle: True
            random_state: 90

          tuner:
            search_algorithm:
              module: ray.tune.search.optuna.OptunaSearch
              params:
            tune_config:
              num_samples: 5
              metric: root_mean_square_error
              mode: min
            run_config:
              stop:
                training_iteration: 2
            scheduler:
              module: ray.tune.schedulers.HyperBandScheduler
              params:

        load:
          save_best_config_params: True

Transform
---------

This phase runs an hyperparameter search algorithm for each model defined in
pipeline file. Furthermore, the user can define a set of metrics to evaluate
the experiments, the model's hyperparamaters to tune, the strategy to split
train and test data and parameters of search algorithm.

The available configurations are the following:

- **metrics** (list, optional): a list of metrics to evaluate the models. Any
  metric that it exists in ``sklearn.metrics`` is allowed. Default values are
  ``mean_squared_error``, ``mean_absolute_percentage_error``,
  ``median_absolute_error``, ``r2_score``, ``mean_absolute_error``,
  ``root_mean_square_error``.
- **models** (list[dicts]): a list of models to search best configuration.
  For each model, it specifies the ``module``
  e.g. ``sklearn.ensemble.RandomForestRegressor`` and the ``search_space``.
  The **search space** is a dictionary with the model's hyperparamater.
  For each hyperparamater to tune, it defines the ``method`` e.g. ``randint``
  to apply and ``values`` e.g. ``2, 110``.
- **cross_validation** (dict, optional): defines which cross-validation
  strategy to use for training each model. Valid values: ``k_fold``,
  ``repeated_k_fold``, ``shuffle_split``, ``leave_one_out``.
  Default: ``k_fold``.
- **tuner** (dict): defines the configuration of tune process.
  The search algorithm is defined in ``search_algorithm`` key e.g.
  ``ray.tune.search.optuna.OptunaSearch``. Also, it is possible to specify
  parameters of algorithms in ``params`` key. The ``tune_config`` defines the
  **metric** to optimize. Furthermore, the ``mode`` allows to define the way
  to optimize the metric. The valid values are ``max`` or ``min``.
  The ``run_config`` defines different parameters during the search.
  For example with ``stop`` it is possible to specify the iterations of
  training step. Finally, the ``scheduler`` allows to define different
  strategies during the search process.

**Note**: As default, the **metrics** list contains only a regression metrics. It
should be pointed out that the metrics depends on **problem_type**.

**Note**: For instance, in **tuner** parameters if **problem type** is
classification and metric is ``accuracy`` the ``mode`` could be ``max``. On
the other hand, if **problem type** is regression and metric is
``root_mean_square_error`` the ``mode`` could be ``min``.

Load
----

In load phase the possible configurations are the following:

- **save_best_config_params** (bool, optional): store a yaml file with best
  model configuration or not. The filename is ``best_config_params.yaml``.

.. _model-step:

Model
=====

This step is responsible for model management.

It is made up for the following ETL phases:

- **extract**: the purpose of this phase is to read a previously saved model.
- **transform**: this phase applies the common model functions:
  training, testing and cross-validation
- **load**: it saves the initialized model.

In addition, there are two new keys:

- **estimator_type**: the kind of estimator: regressor or classifier.
- **estimator_config**: an specific estimator configuration to use.

The following example shows all keys that can be specified in a pipeline
file:

.. code:: yaml

   model:
       estimator_type: regressor
       estimator_config:
           module: sklearn.ensemble.RandomForestRegressor
           hyperparameters:
               n_estimators: 100

       extract:
         filepath: models/sklearn.regressor.20220819-122417.sav

       transform:
         fit:
           cross_validation:
             strategy: k_fold
             n_splits: 10
             shuffle: True
             random_state: 90
         predict:
           path: data/processed

       load:
         path: data/models/

Estimator config
----------------

The **estimator_config** is an optional key that allows to specify the
estimator and its hyperparameters.

**Note**: if a **Benchmark Step** runs before the model step, the best
estimator will be selected and the **estimator_config** will be ignored.

**Note**: if there is not a **Benchmark Step** and the **estimator_config**
is not specified, a default model will be used.

Extract
-------

In extract phase the possible configurations are the following:

- **filepath** (str, optional): file path of model to read. It is an
  optional parameter with default value:
  ``models/sklearn.regressor.20220819-122417.sav``.

**Note**: the framework only allows to extract models generated by the
framework which follow the filename convention
``{model_type}.{estimator_type}.{datetime}.sav``

Transform
---------

This phase applies the common model functions: fit, predict and
cross-validation. The available configurations are the following:

- **fit** (dict): requests a model training on the current dataset.
- **cross_validation** (dict, optional): requests to cross-validate the
  model. At the end, the model will be trained on the whole dataset.
- **strategy** (str, optional): the strategy to use to make the partition
  of the data. Valid values: ``k_fold``, ``repeated_k_fold``,
  ``shuffle_split``, ``leave_one_out``. Default: ``k_fold``.
- **kwargs**: available parameters for the sklearn cross-validation strategy
  selected.
- **predict** (dict): requests to run predictions over the dataset.
- **path** (str, optional): the directory where the predictions will be
  stored. Default value: ``data/processed``.

**Note**: When specifying **transform** in this step, at least **fit**
or **predict** should be set. Otherwise, the **transform** phase will be
ignored.

**Note**: Specifying **fit** and **predict** in the same pipeline,
assuming only one data step has run, the predictions will be generated
over the same dataset where the model has been trained.

Load
----

In load phase the possible configurations are the following:

- **path** (str, optional): the directory where the model will be saved.

**Note**: the filename is generated by the framework following the
following convention: ``{model_type}.{estimator_type}.{datetime}.sav``

