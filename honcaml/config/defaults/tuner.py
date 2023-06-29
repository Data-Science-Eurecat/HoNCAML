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
