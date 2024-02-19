=======
 Usage
=======

Setup
=====

To set up and install HoNCAML, just run the following:

.. code-block:: console

   $ make all

This will do the following:

- Create a virtual environment to not interfere with the current environment
- Install the library and its dependencies
- Generate documentation

Virtual environment directory is located in **./venv** by default, but it can
be changed by changing the variable *ENV_PATH* located in **Makefile**.

Quick execution
===============

Example data
------------

For a quick usage with example data and configuration, just run:

.. code-block:: console

   honcaml -e {example_directory}

This would create a directory containing sample data and configuration to see
how HoNCAML works in a straightforward manner. Just enter the specified
directory: `cd {example_directory}` and run one of the pipelines located in
*files* directory. For example, a benchmark for a classification task:

.. code-block:: console

   honcaml -c files/classification_benchmark.yaml

Custom data
-----------

For a quick train execution, given that a dataset is available with the target
value informed, it is necessary to first create a basic configuration file:

.. code-block:: console

   honcaml -b {config_file} -t {pipeline_type}

Being ``{config_file}`` the path to the file containing the configuration in
yaml extension, and being ``{pipeline_type}`` one of the supported: train, predict
or benchmark.

The specified keys of the file should be filled in, and afterwards it is
possible to run the intended pipeline with the following command:

.. code-block:: console

   honcaml -c {config_file}

This will run the pipeline and export the trained model.

Detailed configuration
======================

In the case of advanced configuration, there is the option of generating a more
complete one, instead of the basic mentioned above:

.. code-block:: console

   (.venv) $ honcaml -a {config_file} -t {pipeline_type}

Advanced configuration files contain comments with required information to fill
in the blanks. All the details of the configuration file are explained in
:ref:`configuration`. Moreover, many examples can be found at
*honcaml/config/examples*.

Executing from the GUI
======================

To run the HoNCAML GUI locally in a web browser tab, run the following command:

.. code-block:: console

   (.venv) $ honcaml -g

It allows to execute HoNCAML providing a datafile and a configuration file, or
to manually select the configuration options instead of providing the file.

When using the manual configuration, it allows both levels of configuration:
Basic, for a faster execution, and Advanced, allows users to configure the
model hyperparameters; and three functionalities: Benchmark, Train and Predict.

Command-line reference
======================

The command-line reference usage is the following:

.. code-block:: console

   usage: honcaml [<args>]
    options:
  -h, --help            show this help message and exit
  -v, --version         HoNCAML current version
  -c CONFIG, --config CONFIG
                        Run HoNCAML through a configuration file.
                        The file specifies which pipeline/s to run and their parameters
  -e EXAMPLE, --example EXAMPLE
                        Store example data with configuration to the specified directory
  -l LOG, --log LOG     file path in which to store execution log
  -b GENERATE_BASIC_CONFIG, --generate-basic-config GENERATE_BASIC_CONFIG
                        generate most basic YAML configuration file. Requires -t argument
  -a GENERATE_ADVANCED_CONFIG, --generate-advanced-config GENERATE_ADVANCED_CONFIG
                        generate advanced YAML configuration file. Requires -t argument
  -t {train,predict,benchmark}, --pipeline-type {train,predict,benchmark}
                        type of execution used while creating YAML configuration.
                        Only makes sense together with -a or -b arguments
  -g, --gui             open GUI in a web browser tab

Extending HoNCAML
=================

It is even possible to further extend or optimize HoNCAML tweaking its
internals. Details on how to do this are explained in :ref:`reference`.
