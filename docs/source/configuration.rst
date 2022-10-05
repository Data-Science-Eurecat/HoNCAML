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

The configuration should be done through a `YAML <https://yaml.org/spec/>`_
file, which contains all the pipeline and steps options that are considered.

The steps configuration are detailed below.

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

When **extract** phase is empty, the framework reads the file
``data/raw/dataset.csv``. Furthermore, it gets ``target`` column name as a
model target. In addition, it uses all columns as features.

In the example below, the framework reads dataset from
``data/raw/boston_dataset.csv``. Also, it gets ``price`` column as a target.
Finally, it uses all columns as features.

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

Estimator Type
--------------

This is a mandatory key that identifies the kind of problem and allows
to select the kind of model to use. Valid values are: ``classifier``,
``regressor``.

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

