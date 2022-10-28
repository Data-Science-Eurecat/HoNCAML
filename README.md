# HoNCAML

Description
-----------

HoNCAML (Holistic No Code Automated Machine Learning) is a tool aimed to run
automated machine learning pipelines for problems of diferent nature; main
types of pipeline would be:

1. Training the best possible model for the problem at hand
2. Use this model to predict other instances

At this moment, the following types of problems are supported:

- Regression
- Classification

Quick usage
-----------

To use HoNCAML, first install it from source, together with its documentation:

   ```commandline
   make install doc
   ```

For a quick train execution, given that a dataset is available with the target
value informed, it is necessary to first create a basic configuration file:

   ```commandline
   honcaml -b {config_file}
   ```

Being ``{config_file}`` the path to the file containing the configuration.

The specified keys of the file should be filled in, and afterwards it is
possible to run the intended pipeline with the following command:

   ```commandline
   honcaml -c {config_file}
   ```

This will run the pipeline and export the trained model.
