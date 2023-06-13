default_tuner = {
    "search_algorithm": {
        "module": "ray.tune.search.optuna.OptunaSearch",
        "params": None
    },
    "tune_config": {
        "time_budget_s": 10,
        "num_samples": 2,
        "metric": {
            "regression": "root_mean_square_error",
            "classification": "accuracy"
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
