=======
 Usage
=======

Right now, HONCAML covers the following

Installation
============

To use HONCAML, first install it using pip:

.. code-block:: console

   (.venv) $ pip install honcaml

Main execution
===============

Train
-----

For a quick execution, given that a configuration file has been filled and the
dataset is available with the target value informed, it is just necessary to
execute honcaml directly:

.. code-block:: console
             
   (.venv) $ honcaml -c {config_file}

Being ``{config_file}`` the path to the file containing the configuration.

This will run the pipeline and export the results as specified.

The command-line reference usage is the following:

.. code-block:: console

   usage: honcaml [<args>]
    options:
    -h, --help            show this help message and exit
    -v, --version         HONCAML current version
    -c CONFIG, --config CONFIG
    YAML configuration file specifying pipeline options
    -l LOG, --log LOG     File path in which to store execution log

Detailed configuration
======================

It is expected to configure the pipeline prior to its execution. All the
details are explained in :ref:`configuration`.

Extending HONCAML
=================

It is even possible to further extend or optimize HONCAML tweaking its
internals. Details on how to do this are explained in :ref:`reference`.
