# Paper

## Introduction

This is the code to reproduce the quantitative benchmark detailed in the
HoNCAML paper.

## Requirements

The following stack is required to perform the benchmark:

- GNU/Linux machine with utilities: bash, make, wget, head, grep, sed
- Python3.10 with venv module
- Python3.9 with venv module[^fn1]
- Java[^fn2]

[^fn1]: Required by auto-sklearn and autoPyTorch frameworks
[^fn2]: Required by H2O framework

## Design

Although several metrics are computed for both classification and regression
problems, due to both clarity and framework limitations, the metric to be
optimized is a single one:

- For classification problems, the metric to be optimized is: *f1-score macro*.

- For regression problems, the metric to be optimized is: *mean absolute error*.

The benchmark is designed to take into account variability within a same
framework when getting results. That is why not only several datasets are taken
into account for the process, but for each of them there are several splits
generated. Therefore, when obtaining results, both mean and variance of the
metrics will be computed.

## Execution

In order to execute the benchmark, it is just required to run the following
from within the *paper* directory:

```sh
make all
```

This will do the following:

- Create a global python virtual environment for main scripts
- Download all datasets from OpenML platform, in ARFF format
- Convert all datasets to standard CSV format
- Generate dataset splits
- Execute all frameworks for all datasets. This means, for each split:
  - Search for best model using the framework
  - Use the best model to predict values
- Generate metrics for all frameworks and all datasets

## Considerations

There are several considerations regarding the benchmark process:

- It is expected to last 30 hours approximately, if done sequentially.
- It is designed in a way that avoids nested loopings in computing intensive
  tasks (search for best model), leveraging /make/ tool.
- Related to the previous point, if the process is stopped at any point,
  already generated files (both intermediate and final results) are not
  generated again.
- Frameworks and datasets have been selected with a specific criteria detailed
  in the paper, but the code is designed to be extended if needed.
- There are frameworks with specific requirements in terms of input data;
  therefore, different preprocessing steps may apply to different
  frameworks. This is detailed again in the paper.
- In the long term, the objective is to include HoNCAML in the [AMLB
  benchmark](https://openml.github.io/automlbenchmark/index.html).
- Not all frameworks ensure full reproducibility, as they do not provide a way
  to set a seed during the search of the best model.
- To run the benchmark considering a subset of datasets and/or frameworks, it
  is just a matter of updating the **DATASETS** and/or **FRAMEWORKS** variables
  at the beginning of the *Makefile*.
