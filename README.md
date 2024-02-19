# HoNCAML

HoNCAML (Holistic No Code Automated Machine Learning) is a tool aimed to run
automated machine learning pipelines for problems of different nature; main
types of pipeline would be:

1. Training the best possible model for the problem at hand
2. Use this model to predict other instances

## Why HoNCAML

### Focus

HoNCAML has been designed having the following aspects in mind:

* Ease of use
* Modularity
* Extensibility
* Simpler is better

### Users

There are (at least) two main types of users who could benefit from this tool:

1. **Regular users**: In terms of programming experience and/or machine learning
   knowledge. It would be possible for them to get results in an easy way.
2. **Advanced users**: It is possible to customise experiments in order to
   adapt to a specific use case that a user with previous knowledge would like.

### Pipelines

This library assumes data has tabular format, and is clean enough to be used to
train models.

At this moment, the following types of problems are supported:

* Regression
* Classification

Regarding available models, the following are supported:

* Sklearn models
* Pytorch (neural net) models

However, due to its nature, extend the library to include other type of
problems and models should be not only feasible, but intuitive.

## Installation

To set up and install HoNCAML, just run the following within a virtual
environment:

   ```commandline
   make install
   ```
Virtual environment directory is located in **./venv** by default, but it can
be changed by changing the variable *ENV_PATH* located in **Makefile**.

## Quick execution

### Example data

For a quick usage with example data and configuration, just run:

   ```commandline
   honcaml -e {example_directory}
   ```

This would create a directory containing sample data and configuration to see
how HoNCAML works in a straightforward manner. Just enter the specified
directory: `cd {example_directory}` and run one of the pipelines located in
*files* directory. For example, a benchmark for a classification task:

   ```commandline
   honcaml -c files/classification_benchmark.yaml
   ```

### Custom data

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

## Detailed configuration

In the case of advanced configuration, there is the option of generating a more
complete one, instead of the basic mentioned above:

```commandline
   honcaml -a {config_file} -t {pipeline_type}
```

Advanced configuration files contain comments with required information to fill
in the blanks. All the details of the configuration file are explained in
the documentation. Moreover, many examples can be found at
[examples](honcaml/config/examples).

## Executing from the GUI

To run the HoNCAML GUI locally in a web browser tab, run the following command:

   ```commnadline
   honcaml -g
   ```

It allows to execute HoNCAML providing a datafile and a configuration file, or
to manually select the configuration options instead of providing the file.

When using the manual configuration, it allows both levels of configuration:
Basic, for a faster execution, and Advanced, allows users to configure the
model hyperparameters; and three functionalities: Benchmark, Train and Predict.

## Contribute

All contributions are more than welcome! For further information, please refer
to the [contribution documentation](CONTRIBUTING.md).

# Bugs

If you find any bug, please report it as an issue.

# Contact

Should you have any inquiry regarding the library or its development, please
contact the [Applied Machine Learning team](mailto:aml@eurecat.org).
