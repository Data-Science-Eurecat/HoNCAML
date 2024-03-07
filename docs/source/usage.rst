=======
 Usage
=======

Requirements
============

To use HoNCAML, it is required to have Python >= 3.10.

Install
=======

To install HoNCAML, run: `pip install honcaml`

Command line execution
======================

Quick execution with example data
---------------------------------

For a quick usage with example data and configuration, just run:

.. code:: console

   honcaml -e {example_directory}

This would create a directory containing sample data and configuration
to see how HoNCAML works in a straightforward manner. Just enter the
specified directory: ``cd {example_directory}`` and run one of the
pipelines located in *files* directory. For example, a benchmark for a
classification task:

.. code:: console

   honcaml -c files/classification_benchmark.yaml

Standard execution
------------------

To start a HoNCAML execution for a particular pipeline, first it is
needed to generate the configuration file for it. It may be easy to
start with a template, which is provided by the CLI itself.

In case a basic configuration file is enough, with the minimum required
options, the following should be invoked:

.. code:: console

   honcaml -b {config_file} -t {pipeline_type}

On the other hand, there is the possibility of generating an advanced
configuration file, with all the supported options:

.. code:: console

   honcaml -a {config_file} -t {pipeline_type}

In both cases, ``{config_file}`` should be a path to the file containing
the configuration in yaml extension, and ``{pipeline_type}`` one of the
supported: train, predict or benchmark.

When having a filled configuration file to run the pipeline, it is just
a matter of executing it:

.. code:: console

   honcaml -c {config_file}

For example, the following basic configuration would train a default model
for classification and store it.

.. code:: yaml

    global:
      problem_type: classification

    steps:
      data:
        extract:
          filepath: data/dataset.csv
          target: class
        transform:

      model:
        transform:
          fit:
        load:
          filepath: default_model.sav

GUI execution
=============

To run the HoNCAML GUI locally in a web browser tab, run the following
command:

.. code:: console

   honcaml -g

It allows to execute HoNCAML by interactively selecting pipeline
options, although it is possible to run a pipeline by uploading its
configuration file as well.
