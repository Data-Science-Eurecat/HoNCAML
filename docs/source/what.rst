=================
 What is HoNCAML
=================

HoNCAML (Holistic No Code Automated Machine Learning) is a tool aimed to
run automated machine learning pipelines, and specifically focused on
finding the best model and hyperparameters for the problem at hand.

Following the `no code
paradigm <https://en.wikipedia.org/wiki/No-code_development_platform>`_,
no Python knowledge is needed. There are two ways to define pipelines:

* Through the Graphical User Interface
* Through `YAML <https://yaml.org/>`_ configuration files

HoNCAML (Holistic No Code Automated Machine Learning) is a tool aimed to run
automated machine learning pipelines, and specifically focused on finding the
best model and hyperparameters for the problem at hand.

Pipelines
=========

There are three types of provided pipelines.

Train
-----

Train a specific model with the hyperparameters specified.

- Input: A dataset for the training.
- Output: The model object stored to disk.

Predict
-------

Use a model to generate predictions for a specific dataset.

- Input: A dataset for the test, together with a model object.
- Output: A tabular file with the predictions.

Benchmark
---------

Search for the best model and hyperparameters for the dataset at hand.

- Input: A dataset for the benchmark.
- Output: Main output is a configuration file with the best model and
  hyperparameters, and a tabular file with the results for all configurations
  tested.

Focus
=====

HoNCAML has been designed having the following aspects in mind:

* Ease of use
* Modularity
* Extensibility

Users
=====

HoNCAML does not assume any kind of technical knowledge, but at the same time
it is designed to be extended by expert people. Therefore, its user base may
range from:

* **Basic users**: In terms of programming experience and/or machine learning
  knowledge. It would be possible for them to get results in an easy way.

* **Advanced users**: It is possible to customize experiments in order to
  adapt to a specific use case that may be needed by an expert person.

Support
=======

Regarding each of the following concepts, HoNCAML supports specific sets
of them; nevertheless, due to its nature, extend the library further
should be not only feasible, but intuitive.

Data structure
--------------

For now only data with tabular format is supported. However, HoNCAML
provides special preprocessing methods if needed:

* Normalization
* One hot encoding of categorical features

Problem type
------------

At this moment, the following types of problems are supported:

* Regression
* Classification

Model type
----------

Regarding available models, the following are supported:

* Sklearn models (ML)
* Pytorch models (DL)
