# Split options
NUM_SPLITS = 5
RANDOM_SEED = 42
TEST_PCT_SPLIT = 0.25

# Dataset options
DATASET_OPTIONS = {
    "abalone": {
        "type": "regression", "target": "rings"
    },
    "diamonds": {
        "type": "regression", "target": "price"
    },
    "house_sales": {
        "type": "regression", "target": "price"
    },
    "sensory": {
        "type": "regression", "target": "Score"
    },
    "Moneyball": {
        "type": "regression", "target": "RS"
    },
    "OnlineNewsPopularity": {
        "type": "regression", "target": "shares"
    },
    "german_credit": {
        "type": "classification", "target": "class"
    },
    "diabetes": {
        "type": "classification", "target": "class"
    },
    "ada": {
        "type": "classification", "target": "class"
    },
    "ozone-level-8hr": {
        "type": "classification", "target": "Class"
    },
    "Australian": {
        "type": "classification", "target": "A15"
    },
    "adult": {
        "type": "classification", "target": "class"
    },
}

# Parameters for benchmark
BENCHMARK_OPTIONS = {
    'max_seconds': 180,
    'seed': 20
}

# Metrics details
METRICS = {
    'regression': [
        ('mean_absolute_percentage_error', {}),
        ('mean_squared_error', {})
    ],
    'classification': [
        ('f1_score', {'average': 'macro', 'zero_division': 1}),
        ('recall_score', {'average': 'macro', 'zero_division': 1}),
        ('precision_score', {'average': 'macro', 'zero_division': 1}),
        ('roc_auc_score', {})
    ]
}
