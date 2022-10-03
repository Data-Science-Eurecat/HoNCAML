=======
 Usage
=======

Installation
============

To use HONCAML, first install it using pip:

.. code-block:: console

   (.venv) $ pip install honcaml

Quick execution
===============

For a quick execution, a dataset is available, it is just necessary to execute
honcaml directly:

.. code-block:: console
             
   (.venv) $ python -m honcaml -i {dataset}

Being ``{dataset}`` the path to the file containing the dataset.

This will run the pipeline and export the results in the standard path.

Detailed configuration
======================

For advanced users, it is expected to configure the pipeline prior to its
execution. All the details are explained in :ref:`configuration`.

Extending HONCAML
=================

It is even possible to further extend or optimize HONCAML tweaking its
internals. Details on how to do this are explained in :ref:`reference`.
