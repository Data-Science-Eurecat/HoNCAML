default_model_step = {
    "extract": {
        "filepath": "models/sklearn.regressor.20220819-122417.sav"
    },
    "transform": {
        "default_estimator": {
            "regression": {
                "module": "sklearn.ensemble.RandomForestRegressor",
                "hyper_parameters": {
                    "n_estimators": 100
                }
            },
            "classification": {
                "module": "sklearn.ensemble.RandomForestClassifier",
                "hyper_parameters": {
                    "n_estimators": 100
                }
            }
        },
        "fit": {
            "cross_validation": {
                "strategy": "k_fold",
                "n_splits": 3,
                "shuffle": True,
                "random_state": 90
            }
        },
        "predict": {
            "path": "data/processed"
        }
    },
    "load": {
        "path": "data/models/"
    }
}
