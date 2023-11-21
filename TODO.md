# Development

HoNCAML is in ongoing development, and it is easier for us to have a central
place where to store planned features or changes to the library.

## Long term

As a long term development goals, we would like to support the following:

- Data beyond tabular, for example:
  - Unstructured text
  - Images

- New problem types, for example:
  - Time series problems
  - Clustering
  
## Internal features

Features that imply improvements to the current structure are the following:

### Include Neural Architecture Search in GUI (torch models)

Right now, GUI only supports sklearn-based models; it should be great to be
able to consider NAS problems with torch models as well

### Create class to manage parameters

Ideally, there would be a class that manages input parameter configuration,
which would include the following:

- User and default parameters as attributes
- Merge function that outputs definite configuration
- Validator (cerberus) that handles configuration validation after merge

### Add model evaluation with test data

In data step, add `validate_size` option and store the specified percentage to
evaluate the model at the end.

### Process data refactoring

Right now, data processing (normalization) is correctly done in sklearn models
only. The way to support torch models without doing a custom implementation
would be the following:

- Create a Processing class that handles all pre and postprocessing of data,
  that will generalize current Normalization method.
- This class will be instantiated before and after the model call, within the
  model object, in order to be included when storing it.

### Save model objects with custom names

Right now, it is not possible to properly specify custom names to save
models.

## Fixes

There are still some pending fixes that should be relatively straightforward:

### GUI: Add button to store trained model

Add this functionality within `download_trained_model_button` function.

### GUI: Add custom elements to multi-selects

Add them within `benchmark_model_params_configs` function.

### GUI: Configure other CV strategies than K-Folds

Only k-fold validation is configured in GUI, in `cross_validation_configs`.

### GUI: Read train results from file if exists

If train results file has been stored, use information from this file in
`display_results`.

### Improve code coverage

Code coverage can always be improved. It is only a matter of running: `make
tests` and see room for improvement in there.
