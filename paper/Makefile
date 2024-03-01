# Datasets and frameworks to consider
DATASETS = abalone diamonds german_credit house_sales sensory diabetes \
			ada Australian ozone-level-8hr adult Moneyball \
			OnlineNewsPopularity
FRAMEWORKS = autogluon autokeras autopytorch autosklearn flaml h2o honcaml lightautoml tpot

# Specify paths
GLOBAL_ENV_PATH = .venv
ENV_PATH = environments
REQ_PATH = requirements
RAW_PATH = data/raw
CONVERT_PATH = data/processed
SPLITS_PATH = data/splits
PREDICTIONS_PATH = predictions
RESULTS_PATH = results
FRAMEWORKS_PATH = src/frameworks

# Generate auxiliary variables from defined ones
raw_files = $(foreach dataset,$(DATASETS),$(RAW_PATH)/$(dataset).arff)
convert_files = $(foreach dataset,$(DATASETS),$(CONVERT_PATH)/$(dataset).csv)
split_files = $(foreach dataset,$(DATASETS),$(SPLITS_PATH)/$(dataset).csv)
framework_envs = $(foreach framework,$(FRAMEWORKS),$(ENV_PATH)/$(framework)/pyvenv.cfg)
prediction_files = \
	$(foreach framework,$(FRAMEWORKS), \
		$(foreach dataset,$(DATASETS), \
			$(PREDICTIONS_PATH)/$(framework)/$(dataset).csv))
result_files = $(foreach framework,$(FRAMEWORKS),$(RESULTS_PATH)/$(framework).csv)

## Specify targets
.PHONY: global_env download convert split envs execute result clean

# Default target: set up, download and convert data, and run benchmark
all: global_env download convert split envs execute result

global_env: .venv/pyvenv.cfg

.venv/pyvenv.cfg: requirements.txt
	python3 -m venv $(GLOBAL_ENV_PATH)
	$(GLOBAL_ENV_PATH)/bin/pip install -r requirements.txt

# Download all files
download: $(raw_files)
# Rule to download file by file
$(RAW_PATH)/%.arff:
	./src/download $@

# Convert all files
convert: $(convert_files)
# Rule to convert file by file
$(CONVERT_PATH)/%.csv: $(RAW_PATH)/%.arff
	$(GLOBAL_ENV_PATH)/bin/python -m src.convert "$<" "$@"
	./src/modify $@

# Split all files
split: $(split_files)
# Rule to generate splits file by file
$(SPLITS_PATH)/%.csv: $(CONVERT_PATH)/%.csv
	$(GLOBAL_ENV_PATH)/bin/python -m src.generate_splits "$<" "$@"

# Generate all environments
envs: $(framework_envs)
# Rule to generate the environment for each framework
$(ENV_PATH)/%/pyvenv.cfg: $(REQ_PATH)/%.txt
	./src/create_environment $* $(ENV_PATH) $(REQ_PATH)

execute: $(prediction_files)
# Generate dinamically rules for each combination of dataset and framework
define PREDICT_FRAMEWORK_DATASET_RULE
$(PREDICTIONS_PATH)/$(1)/$(2).csv: $(SPLITS_PATH)/$(2).csv | $(PREDICTIONS_PATH)/$(1)
	${ENV_PATH}/$(1)/bin/python -m src.run_framework \
		$(1) $(SPLITS_PATH)/$(2).csv $(PREDICTIONS_PATH)/$(1)/$(2).csv
endef
# Effectively call the aforementioned rule
$(foreach framework, $(FRAMEWORKS), \
	$(foreach dataset, $(DATASETS), \
			$(eval $(call PREDICT_FRAMEWORK_DATASET_RULE,$(framework),$(dataset)))))

$(PREDICTIONS_PATH)/%:
	mkdir $(PREDICTIONS_PATH)/$*

# Compute all results
result: $(result_files)
# Rule to compute results for each framework
$(RESULTS_PATH)/%.csv: $(PREDICTIONS_PATH)/%
	${GLOBAL_ENV_PATH}/bin/python -m src.compute_metrics \
		$(PREDICTIONS_PATH)/$* $(RESULTS_PATH)/$*.csv

# Clean all information
clean:
	rm -rf $(ENV_PATH)/*
	rm -rf $(RAW_PATH)/*.arff
	rm -rf $(CONVERT_PATH)/*.csv
	rm -rf $(SPLITS_PATH)/*.csv
	rm -rf $(PREDICTIONS_PATH)/*/*.csv
	rm -rf $(RESULTS_PATH)/*.csv
