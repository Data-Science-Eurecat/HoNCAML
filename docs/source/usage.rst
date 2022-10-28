=======
 Usage
=======

Right now, HoNCAML covers the following

Installation
============

To use HoNCAML, first install it from source:

.. code-block:: console

   (.venv) $ make install

Quick execution
===============

Train
-----

For a quick execution, given that a dataset is available with the target value
informed, it is necessary to first create a basic configuration file:

.. code-block:: console
             
   (.venv) $ honcaml -b {config_file}

Being ``{config_file}`` the path to the file containing the configuration.

The specified keys of the file should be filled in, and afterwards it is
possible to run the intended pipeline with the following command:

.. code-block:: console
             
   (.venv) $ honcaml -c {config_file}

This will run the pipeline and export the trained model.

Detailed configuration
======================

In the case of advanced configuration, there is the option of generating a more
complete one, instead of the basic mentioned above:

.. code-block:: console
             
   (.venv) $ honcaml -a {config_file}

In this case, there are default values speficied which should be replaced. In
general, all the details of the configuration file are explained in
:ref:`configuration`.

Command-line reference
======================

The command-line reference usage is the following:

.. code-block:: console

   usage: honcaml [<args>]
    options:
    -h, --help            show this help message and exit
    -v, --version         HoNCAML current version
    -c CONFIG, --config CONFIG
    YAML configuration file specifying pipeline options
    -l LOG, --log LOG     File path in which to store execution log
    -b GENERATE_BASIC_CONFIG, --generate-basic-config GENERATE_BASIC_CONFIG
                        Generate most basic YAML configuration file
    -a GENERATE_ADVANCED_CONFIG, --generate-advanced-config GENERATE_ADVANCED_CONFIG
                        Generate advanced YAML configuration file

Extending HoNCAML
=================

It is even possible to further extend or optimize HoNCAML tweaking its
internals. Details on how to do this are explained in :ref:`reference`.
