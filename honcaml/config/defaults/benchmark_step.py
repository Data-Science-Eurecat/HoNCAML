from honcaml.config.defaults.search_spaces import default_search_spaces
from honcaml.config.defaults.tuner import default_tuner


default_benchmark_step = {
    # Read a previously saved learner
    "extract": None,
    "transform": {
        "models": default_search_spaces,
        "cross_validation": {
            "module": "sklearn.model_selection.KFold",
            "params": {"n_splits": 3}
        },
        "tuner": default_tuner,
    },
    # Save the learner to disk and the results
    "load": {
        "path": "honcaml_reports",
        'save_best_config_params': True
    }
}
