# Configuration options

params = {

    # Global paths
    "paths": {
        "metrics_folder": "honcaml_reports"
    },

    # Logging options
    "logging": {
        "formatter": {
            "format": "%(asctime)s - %(message)s",
            "time_format": "%Y-%m-%d %H:%M:%S"
        },
        "level": "DEBUG",
    },

    # Pipeline validation rules
    "pipeline_rules": None,

    "step_rules": {
        "data": {
            "extract": {
                "filepath": {
                    "required": True
                },
                "target": {
                    "required": True
                }
            },
            "transform": {
                "normalize": {
                    "features": {
                        "module": {
                            "required": True
                        },
                        "target": {
                            "module": {
                                "required": True
                            }
                        }
                    }
                }
            },
            "load": {
                "path": {
                    "required": True
                }
            }
        },
        "model": {
            "extract": {
                "filepath": {
                    "required": True
                }
            },
            "transform": {
                "fit": {
                    "cross_validation": {
                        "strategy": {
                            "required": True
                        }
                    }
                },
                "predict": {
                    "path": {
                        "required": True
                    }
                }
            },
            "load": {
                "path": {
                    "required": True
                }
            }
        },
        "benchmark": None
    },

    # Default estimators
    "default_regressor_estimator": {
        "module": "sklearn.ensemble.RandomForestRegressor",
        "hyper_parameters": {
            "n_estimators": 100
        }
    },

    # Default pipeline steps settings
    "pipeline_steps": {
        # Data Step settings
        "data": {
            "extract": {
                "filepath": "data/raw/dataset.csv"
            },
            "transform": {
                "normalize": {
                    "features": {
                        "module": "sklearn.preprocessing.StandardScaler"
                    },
                    "target": {
                        "module": "sklearn.preprocessing.StandardScaler"
                    }
                }
            }
        },
        # Model Step settings
        "model": {
            "extract": {
                "filepath": "models/sklearn.regressor.20220819-122417.sav"
            },
            "transform": {
                "fit": {
                    "cross_validation": {
                        "strategy": "k_fold",
                        "n_splits": 10,
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
        },

        # Benchmark Step settings
        "benchmark": {
            # Read a previously saved learner
            "extract": None,
            "transform": {
                "metrics": [
                    "mean_squared_error",
                    "mean_absolute_percentage_error",
                    "median_absolute_error",
                    "r2_score",
                    "mean_absolute_error",
                    "root_mean_square_error"
                ],
                "tuner": {
                    "algorithm": "ray.tune.search.optuna.OptunaSearch",
                    "models": {
                        "module": "sklearn.ensemble.RandomForestRegressor",
                        "search_space": {
                            "n_estimators": {
                                "method": "randint",
                                "value": [2, 10]
                            }
                        }
                    },
                    "cross_validation": {
                        "strategy": "k_fold",
                        "n_splits": 2,
                        "shuffle": True,
                        "random_state": 90
                    }
                }
            },
            # Save the learner to disk and the results
            "load": {
                'save_best_config_params': True
            }
        }
    },

    # Available models settings
    # List the models with their available hyperparameters to tune
    # with their limits.
    "models": None

}

models_config = {

    "regressors": {
        "sklearn.ensemble.RandomForestRegressor": {
            "search_space": {
                # A list of hyper parameters to search as default
                "n_estimators": {
                    "method": "randint",
                    "value": [2, 100]
                }
            }
        }
    },

    "classifiers": {
        "sklearn.ensemble.RandomForestClassifier": {
            "search_space": {
                # TODO: a list of hyper parameters to search as default
                "n_estimators": {
                    "method": "randint",
                    "value": [2, 100]
                }
            }
        }
    },

    "search_space_mapping": {
        "randint": "tune.randint",
        "choice": "tune.choice"
        # Adding more functions here
    }

}
