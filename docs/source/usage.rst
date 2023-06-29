=======
 Usage
=======

Setup
============

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

To execute HoNCAML, all commands should be executed from within the generated
virtual environment; to enter it, run:

.. code-block:: console

   $ source {env_path}/bin/activate

Replacing *{env_path}* with the virtual environment path.

Train
-----

For a quick execution, given that a dataset is available with the target value
informed, it is necessary to first create a basic configuration file with a
pipeline type:

.. code-block:: console
             
   (.venv) $ honcaml -b {config_file} -t {pipeline_type}

Being ``{config_file}`` the path to the file containing the configuration in
yaml extension, and being ``{pipeline_type}`` one of the supported: train, predict
or benchmark.

The specified keys of the file should be filled in as specified, and afterwards
it is possible to run the intended pipeline with the following command:

.. code-block:: console
             
   (.venv) $ honcaml -c {config_file}

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

Command-line reference
======================

The command-line reference usage is the following:

.. code-block:: console

   usage: honcaml [<args>]
    options:
    -h, --help          show this help message and exit
    -v, --version       HoNCAML current version
    -c CONFIG, --config CONFIG
                        YAML configuration file specifying pipeline options
    -l LOG, --log LOG   file path in which to store execution log
    -b GENERATE_BASIC_CONFIG, --generate-basic-config GENERATE_BASIC_CONFIG
                        generate most basic YAML configuration file
    -a GENERATE_ADVANCED_CONFIG, --generate-advanced-config GENERATE_ADVANCED_CONFIG
                        generate advanced YAML configuration file
    -t {train,predict,benchmark}, --pipeline-type {train,predict,benchmark}
                        type of execution used while creating YAML
                        configuration. Only makes sense together with
                        -a or -b arguments.                        

Extending HoNCAML
=================

It is even possible to further extend or optimize HoNCAML tweaking its
internals. Details on how to do this are explained in :ref:`reference`.
