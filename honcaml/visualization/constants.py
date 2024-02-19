import datetime
import os

# General execution path
BASE_PATH = "honcaml_execution"

execution_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
exec_path = os.path.join(BASE_PATH, execution_name)

# Specific execution paths
data_file_path = os.path.join(exec_path, "data_file.csv")
data_file_path_config_file = os.path.join(exec_path, "data_file.csv")
config_file_path = os.path.join(exec_path, "config_file.yaml")
benchmark_results_path = os.path.join(exec_path, "honcaml_reports")
predict_results_path = os.path.join(exec_path, "honcaml_reports")
trained_model_file = os.path.join(exec_path, "model.sav")
logs_path = os.path.join(exec_path, "logs")

# Internal templates path
templates_path = "honcaml/config/templates/"

# dictionary containing the display name of the model and the name of the model
# to use in the configs file
names_of_models = {
    "regression": {
        "Linear Regression": "sklearn.linear_model.LinearRegression",
        "Random Forest Regressor": "sklearn.ensemble.RandomForestRegressor",
        "Linear SVR": "sklearn.svm.LinearSVR",
        "K-Neighbors Regressor": "sklearn.neighbors.KNeighborsRegressor",
        "SGD Regressor": "sklearn.linear_model.SGDRegressor",
        "Gradient Boosting Regressor":
            "sklearn.ensemble.GradientBoostingRegressor",
        "Elastic Net": "sklearn.linear_model.ElasticNet",
        "Neural Network": "torch"
    },
    "classification": {
        "Logistic Regression": "sklearn.linear_model.LogisticRegression",
        "Random Forest Classifier": "sklearn.ensemble.RandomForestClassifier",
        "Linear SVC": "sklearn.svm.LinearSVC",
        "K-Neighbors Classifier": "sklearn.neighbors.KNeighborsClassifier",
        "SGD Classifier": "sklearn.linear_model.SGDClassifier",
        "Gradient Boosting Classifier":
            "sklearn.ensemble.GradientBoostingClassifier",
        "Neural Network": "torch"
    }
}


default_models = {
    "regression": ("Linear Regression", "Random Forest Regressor"),
    "classification": ("Logistic Regression", "Random Forest Classifier")
}


model_configs_helper = {
    "uniform": "Sample a float uniformly between min and max values selected "
               "on the slider",
    "quniform": "Sample a float uniformly between min and max values selected "
                "on the slider, rounding to increments of the value selected "
                "on the number input field",
    "loguniform": "Sample a float uniformly in a log order of magnitude "
                  "between min and max values selected on the slider",
    "randint": "Sample a integer uniformly between min (inclusive) and max "
               "(exclusive) values selected on the slider",
    "qrandint": "Sample a random uniformly between min (inclusive) and max "
                "(inclusive (!)) values selected on the slider, rounding to "
                "increments of the value selected on the number input field",
    "choice": "Sample an option uniformly from the specified choices"
}


metrics_mode = {
    "regression": {
        "mean_squared_error": "min",
        "mean_absolute_percentage_error": "min",
        "median_absolute_error": "min",
        "r2_score": "max",
        "mean_absolute_error": "min",
        "root_mean_squared_error": "min"
    },
    "classification": {
        "accuracy_score": "max",
        "precision_score": "max",
        "recall_score": "max",
        "specificity_score": "max",
        "f1_score": "max",
        "roc_auc_score": "max"
    }
}
