# HoNCAML

## Introduction

HoNCAML (Holistic No Code Automated Machine Learning) is a tool aimed to run
automated machine learning pipelines, and specifically focused on finding the
best model and hyperparameters for the problem at hand.

Following the [no code
paradigm](https://en.wikipedia.org/wiki/No-code_development_platform), no
Python knowledge is needed. There are two ways to define pipelines:

* Through the Graphical User Interface
* Through [YAML](https://yaml.org/) configuration files

## Pipelines

There are three types of provided pipelines:

* **Train**: Train a specific model with its hyperparameters given a dataset.
* **Predict**: Given a dataset, use a specific model to predict the outcome.
* **Benchmark**: Given a dataset, search for the best model and hyperparameters.

## Focus

HoNCAML has been designed having the following aspects in mind:

* Ease of use
* Modularity
* Extensibility
* Simpler is better

## Users

HoNCAML does not assume any kind of technical knowledge, but at the same time
it is designed to be extended by expert people. Therefore, its user base may
range from:

* **Basic users**: In terms of programming experience and/or machine learning
  knowledge. It would be possible for them to get results in an easy way.

* **Advanced users**: It is possible to customize experiments in order to
  adapt to a specific use case that may be needed by an expert person.

## Support

Python version should be >= 3.10.

Regarding each of the following concepts, HoNCAML supports specific sets of
them; nevertheless, due to its nature, extend the library further should be not
only feasible, but intuitive.

### Data structure

For now only data with tabular format is supported. However, HoNCAML provides special
preprocessing methods if needed:

* Normalization
* One hot encoding of categorical features

### Problem type

At this moment, the following types of problems are supported:

* Regression
* Classification

### Model type

Regarding available models, the following are supported:

* Sklearn models (ML)
* Pytorch models (DL)

## Install

To install HoNCAML, run: `pip install honcaml`

## Command line execution

### Quick execution with example data

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

### Standard execution

To start a HoNCAML execution for a particular pipeline, first it is needed to
generate the configuration file for it. It may be easy to start with a
template, which is provided by the CLI itself.

In case a basic configuration file is enough, with the minimum required
options, the following should be invoked:

   ```commandline
   honcaml -b {config_file} -t {pipeline_type}
   ```

On the other hand, there is the possibility of generating an advanced
configuration file, with all the supported options:

   ```commandline
   honcaml -a {config_file} -t {pipeline_type}
   ```

In both cases, ``{config_file}`` should be a path to the file containing the
configuration in yaml extension, and ``{pipeline_type}`` one of the supported:
train, predict or benchmark.

When having a filled configuration file to run the pipeline, it is just a
matter of executing it:

   ```commandline
   honcaml -c {config_file}
   ```

Depending on the pipeline type, the output may be a trained model, predictions,
or benchmark results.

## GUI execution

To run the HoNCAML GUI locally in a web browser tab, run the following command:

   ```commnadline
   honcaml -g
   ```

It allows to execute HoNCAML by interactively selecting pipeline options,
although it is possible to run a pipeline by uploading its configuration file
as well.

## Contribute

All contributions are more than welcome! For further information, please refer
to the [contribution
documentation](https://github.com/Data-Science-Eurecat/HoNCAML/blob/main/CONTRIBUTING.md).

## Bugs

If you find any bug, please check if there is any existing
[issues](https://github.com/Data-Science-Eurecat/HoNCAML/issues), and if not,
open a new one with a clear description.

## Contact

Should you have any inquiry regarding the library or its development, please
contact the [Applied Machine Learning team](mailto:aml@eurecat.org).
