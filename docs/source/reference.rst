.. _reference:

===========
 Reference
===========

HoNCAML follows mainly an
`OOP <https://en.wikipedia.org/wiki/Object-oriented_programming>`_ coding
approach through Python classes. The main ones are detailed in this section.

Execution
=========

The main class used by HoNCAML is execution, which is a wrapper on top of
the :ref:`pipeline` class.

.. autoclass:: honcaml.tools.execution.Execution
   :members:

.. _pipeline:
      
Pipeline
========

A pipeline is made of several :ref:`steps` to be executed.

.. autoclass:: honcaml.tools.pipeline.Pipeline
   :members:

.. _steps:
      
Steps
=====

The step class is the one that determines the parts of a pipeline to run, and
it follows a ETL approach.

.. autoclass:: honcaml.steps.base.BaseStep
   :members:

Data
----

The data step is the one related to data management.

.. autoclass:: honcaml.steps.data.DataStep
   :members:   

It includes the following classes that further configure the step:

- BaseDataset: Defines an abstract class that serves as a parent to the rest
  of the dataset classes (e.g. TabularDataset, etc.)
  
  .. autoclass:: honcaml.data.base.BaseDataset
     :members:

- Normalization: Wraps all normalization methods that apply to the dataset.

  .. autoclass:: honcaml.data.normalization.Normalization
     :members:

- CrossValidationSplit: Applies CV splitting through the dataset.

  .. autoclass:: honcaml.data.transform.CrossValidationSplit
     :members:
               
Model
-----

The model step is the one related to model management.

.. autoclass:: honcaml.steps.model.ModelStep
   :members:

- BaseModel: Defines an abstract class from which models are created.

  .. autoclass:: honcaml.models.base.BaseModel
     :members:
               
               
Benchmark
---------

The benchmark step is the one related to meta-model management, specifically to
select the best model between all of the available options.

.. autoclass:: honcaml.steps.benchmark.BenchmarkStep
   :members:

- BaseBenchmark: Defines an abstract class for model benchmarking.

  .. autoclass:: honcaml.benchmark.base.BaseBenchmark
     :members:      

- EstimatorTrainer: Computes optimised hyperparameters for a specific model,
  based on `tune.Trainable` class.

  .. autoclass:: honcaml.benchmark.trainable.EstimatorTrainer
     :members:

      
