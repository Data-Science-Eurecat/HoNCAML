default_model_step = {
    "extract": {
        "filepath": "models/sklearn.regressor.20220819-122417.sav"
    },
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
            }
        },
        "predict": {
            "path": "data/processed"
        }
    },
    "load": {
        "path": "honcaml_reports"
    }
}
