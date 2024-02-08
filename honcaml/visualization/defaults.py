default_tuner = {
    "search_algorithm": {
        "module": "ray.tune.search.optuna.OptunaSearch",
        "params": None
    },
    "tune_config": {
        "num_samples": 2,
        "metric": {
            "regression": "root_mean_squared_error",
            "classification": "accuracy_score"
        },
        "mode": {
            "regression": "min",
            "classification": "max"
        }
    },
    "run_config": {
        "stop": {
            "training_iteration": 1
        }
    },
    "scheduler": {
        "module": "ray.tune.schedulers.HyperBandScheduler",
        "params": None
    },
}


regression = {
    "sklearn.linear_model.LinearRegression": {
        "fit_intercept": {
            "method": "choice",
            "values": [True, False]
        }
    },
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
            "values": [1, "sqrt", "log2"]
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
            "values": [1.0, "sqrt", "log2"]
        }
    },
    "torch": {
        "epochs": {
            "method": "randint",
            "values": [2, 5]
        },
        "layers": {
            "number_blocks": [3, 6],
            "types": ["Linear + ReLU", "Dropout"],
            "params": {
                "Dropout": {
                    "p": {
                        "method": "uniform",
                        "values": [0.4, 0.6]
                    }
                }
            }
        },
        "loader": {
            "batch_size": {
                "method": "randint",
                "values": [20, 40]
                },
            "shuffle": {
                "method": "choice",
                "values": [True, False]
                }
        },
        "loss": {
            "method": "choice",
            "values": [{"module": "torch.nn.MSELoss"},
                       {"module": "torch.nn.L1Loss"}]
        },
        "optimizer": {
            "method": "choice",
            "values": [
                {
                    "module": "torch.optim.SGD",
                    "params": {
                        "lr": {
                            "method": "loguniform",
                            "values": [0.001, 0.01]
                        },
                        "momentum": {
                            "method": "uniform",
                            "values": [0.5, 1]
                        }
                    }
                },
                {
                    "module": "torch.optim.Adam",
                    "params": {
                        "lr": {
                            "method": "loguniform",
                            "values": [0.001, 0.1]
                        },
                        "eps": {
                            "method": "loguniform",
                            "values": [0.0000001, 0.00001]
                        }
                    }
                }
            ]
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
            "values": [1, "sqrt", "log2"]
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
            "values": [1.0, "sqrt", "log2"]
        }
    },
    "torch": {
        "epochs": {
            "method": "randint",
            "values": [2, 5]
        },
        "layers": {
            "number_blocks": [3, 6],
            "types": ["Linear + ReLU", "Dropout"],
            "params": {
                "Dropout": {
                    "p": {
                        "method": "uniform",
                        "values": [0.4, 0.6]
                    }
                }
            }
        },
        "loader": {
            "batch_size": {
                "method": "randint",
                "values": [20, 40]
                },
            "shuffle": {
                "method": "choice",
                "values": [True, False]
                }
        },
        "loss": {
            "method": "choice",
            "values": [{"module": "torch.nn.CrossEntropyLoss"}]
        },
        "optimizer": {
            "method": "choice",
            "values": [
                {
                    "module": "torch.optim.SGD",
                    "params": {
                        "lr": {
                            "method": "loguniform",
                            "values": [0.001, 0.01]
                        },
                        "momentum": {
                            "method": "uniform",
                            "values": [0.5, 1]
                        }
                    }
                },
                {
                    "module": "torch.optim.Adam",
                    "params": {
                        "lr": {
                            "method": "loguniform",
                            "values": [0.001, 0.1]
                        },
                        "eps": {
                            "method": "loguniform",
                            "values": [0.0000001, 0.00001]
                        }
                    }
                }
            ]
        }
    }
}

default_search_spaces = {
    "regression": regression,
    "classification": classification
}

layers_train_configs = {
    "method": "layers",
    "values": [
        {
            "module": "torch.nn.Linear",
            "params": {"out_features": 64}
        },
        {
            "module": "torch.nn.ReLU"
        },
        {
            "module": "torch.nn.Linear",
            "params": {"out_features": 32}
        },
        {
            "module": "torch.nn.Dropout"
        },
        {
            "module": "torch.nn.Linear"
        }
    ]
}
