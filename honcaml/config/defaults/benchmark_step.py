from honcaml.config.defaults.search_spaces import default_search_spaces
from honcaml.config.defaults.tuner import default_tuner


default_benchmark_step = {
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
        "models": default_search_spaces,
        "cross_validation": {
            "strategy": "k_fold",
            "n_splits": 4,
            "shuffle": True,
            "random_state": 90
        },
        "tuner": default_tuner
    },
    # Save the learner to disk and the results
    "load": {
        'save_best_config_params': True
    }
}
