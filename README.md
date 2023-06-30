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

To set up and install HoNCAML, just run the following:

   ```commandline
   make all
   ```

This will do the following:
- Create a virtual environment to not interfere with the current environment
- Install the library and its dependencies
- Generate documentation

Virtual environment directory is located in **./venv** by default, but it can
be changed by changing the variable *ENV_PATH* located in **Makefile**.

Then, to execute HoNCAML, all commands should be executed from within the
generated virtual environment; to enter it, run:

    ```commandline
    source {env_path}/bin/activate
    ```

Replacing *{env_path}* with the virtual environment path.

For a quick train execution, given that a dataset is available with the target
value informed, it is necessary to first create a basic configuration file:

   ```commandline
   honcaml -b {config_file} -t {pipeline_type}
   ```

Being ``{config_file}`` the path to the file containing the configuration in
yaml extension, and being ``{pipeline_type}`` one of the supported: train, predict
or benchmark.

The specified keys of the file should be filled in, and afterwards it is
possible to run the intended pipeline with the following command:

   ```commandline
   honcaml -c {config_file}
   ```

This will run the pipeline and export the trained model.
