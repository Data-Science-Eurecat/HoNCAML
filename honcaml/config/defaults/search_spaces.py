regression = {
    "sklearn.linear_model.ElasticNet": {
        "l1_ratio": {
            "method": "uniform",
            "values": [0, 1]
        }
    },
    "sklearn.linear_model.SGDRegressor": {
        "loss": {
            "method": "choice",
            "values": ["squared_error", "huber",
                       "epsilon_insensitive", "squared_epsilon_insensitive"]
        },
        "penalty": {
            "method": "choice",
            "values": ["l1", "l2", "elasticnet", None]
        },
        "alpha": {
            "method": "choice",
            "values": [0.0001, 0.001, 0.01]
        }
    },
    "sklearn.svm.LinearSVR": {
        "loss": {
            "method": "choice",
            "values": ["epsilon_insensitive", "squared_epsilon_insensitive"]
        },
        "C": {
            "method": "quniform",
            "values": [0.5, 1, 0.25]
        }
    },
    "sklearn.neighbors.KNeighborsRegressor": {
        "n_neighbors": {
            "method": "randint",
            "values": [3, 20]
        },
        "weights": {
            "method": "choice",
            "values": ["uniform", "distance"]
        },
        "algorithm": {
            "method": "choice",
            "values": ["auto", "ball_tree", "kd_tree"]
        }
    },
    "sklearn.ensemble.RandomForestRegressor": {
        "criterion": {
            "method": "choice",
            "values": ["squared_error", "absolute_error"]
        },
        "n_estimators": {
            "method": "qrandint",
            "values": [20, 140, 40]
        },
        "min_samples_split": {
            "method": "randint",
            "values": [5, 15]
        },
        "max_depth": {
            "method": "qrandint",
            "values": [2, 10, 4]
        },
        "max_features": {
            "method": "choice",
            "values": ["auto", "sqrt"]
        }
    },
    "sklearn.ensemble.GradientBoostingRegressor": {
        "loss": {
            "method": "choice",
            "values": ["squared_error", "absolute_error"]
        },
        "n_estimators": {
            "method": "qrandint",
            "values": [20, 180, 40]
        },
        "min_samples_split": {
            "method": "randint",
            "values": [2, 4]
        },
        "max_depth": {
            "method": "randint",
            "values": [2, 6]
        },
        "max_features": {
            "method": "choice",
            "values": ["auto", "sqrt"]
        }
    }
}

classification = {
    "sklearn.linear_model.LogisticRegression": {
        "penalty": {
            "method": "choice",
            "values": ["elasticnet"]
        },
        "solver": {
            "method": "choice",
            "values": ["saga"],
        },
        "l1_ratio": {
            "method": "uniform",
            "values": [0, 1]
        }
    },
    "sklearn.linear_model.SGDClassifier": {
        "loss": {
            "method": "choice",
            "values": ["squared_error", "huber",
                       "epsilon_insensitive", "squared_epsilon_insensitive"]
        },
        "penalty": {
            "method": "choice",
            "values": ["l1", "l2", "elasticnet", None]
        },
        "alpha": {
            "method": "choice",
            "values": [0.0001, 0.001, 0.01]
        }
    },
    "sklearn.svm.LinearSVC": {
        "loss": {
            "method": "choice",
            "values": ["hinge", "squared_hinge"]
        },
        "C": {
            "method": "quniform",
            "values": [0.5, 1, 0.25]
        }
    },
    "sklearn.neighbors.KNeighborsClassifier": {
        "n_neighbors": {
            "method": "randint",
            "values": [3, 20]
        },
        "weights": {
            "method": "choice",
            "values": ["uniform", "distance"]
        },
        "algorithm": {
            "method": "choice",
            "values": ["auto", "ball_tree", "kd_tree"]
        }
    },
    "sklearn.ensemble.RandomForestClassifier": {
        "criterion": {
            "method": "choice",
            "values": ["gini", "entropy", "log_loss"]
        },
        "n_estimators": {
            "method": "qrandint",
            "values": [20, 180, 40]
        },
        "min_samples_split": {
            "method": "randint",
            "values": [2, 4]
        },
        "max_depth": {
            "method": "choice",
            "values": [2, 4, None]
        },
        "max_features": {
            "method": "choice",
            "values": ["auto", "sqrt", "log2"]
        }
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "loss": {
            "method": "choice",
            "values": ["log_loss", "deviance", "exponential"]
        },
        "n_estimators": {
            "method": "qrandint",
            "values": [20, 180, 40]
        },
        "min_samples_split": {
            "method": "randint",
            "values": [2, 4]
        },
        "max_depth": {
            "method": "randint",
            "values": [2, 6]
        },
        "max_features": {
            "method": "choice",
            "values": ["auto", "sqrt", "log2"]
        }
    }
}

default_search_spaces = {
    "regression": regression,
    "classification": classification
}
