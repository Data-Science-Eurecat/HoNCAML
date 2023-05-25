default_tuner = {
    "search_algorithm": {
        "module": "ray.tune.search.optuna.OptunaSearch",
        "params": None
    },
    "tune_config": {
        "num_samples": 5,
        "metric": "root_mean_square_error",
        "mode": "min"
    },
    "run_config": {
        "stop": {
            "training_iteration": 2
        }
    },
    "scheduler": {
        "module": "ray.tune.schedulers.HyperBandScheduler",
        "params": None
    },
}
