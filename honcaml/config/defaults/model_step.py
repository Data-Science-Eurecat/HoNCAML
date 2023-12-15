default_model_step = {
    "transform": {
        "fit": {
            "estimator": {
                "regression": {
                    "module": "sklearn.ensemble.RandomForestRegressor",
                    "params": {
                        "n_estimators": 100
                    }
                },
                "classification": {
                    "module": "sklearn.ensemble.RandomForestClassifier",
                    "params": {
                        "n_estimators": 100
                    }
                }
            },
            "cross_validation": {
                "module": "sklearn.model_selection.KFold",
                "params": {"n_splits": 3}
            },
            "metrics": {
                "regression": [
                    "mean_squared_error",
                    "mean_absolute_percentage_error",
                    "median_absolute_error",
                    "mean_absolute_error",
                    "root_mean_squared_error"
                ],
                "classification": [
                    "accuracy_score",
                    "precision_score",
                    "recall_score",
                    "specificity_score",
                    "f1_score",
                    "roc_auc_score"
                ]
            }
        },
        "predict": {
            "path": "data/processed"
        }
    },
    "load": {
        "filepath": "data/models/{autogenerate}.sav",
    }
}
