# Configuration options
from honcaml.config.defaults.step_rules import default_step_rules
from honcaml.config.defaults.data_step import default_data_step
from honcaml.config.defaults.model_step import default_model_step
from honcaml.config.defaults.benchmark_step import default_benchmark_step


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
        "level": "INFO",
    },

    # Pipeline validation rules
    "pipeline_rules": None,

    "step_rules": default_step_rules,

    # Default pipeline steps settings
    "pipeline_steps": {
        "data": default_data_step,
        # Model Step settings
        "model": default_model_step,

        # Benchmark Step settings
        "benchmark": default_benchmark_step
    },
}
