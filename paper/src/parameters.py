# Split options
NUM_SPLITS = 3
RANDOM_SEEDS = [16, 30, 42, 65, 87]
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
    'max_seconds': 600,
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

PLOT_OPTIONS = {
    'n_cols': 3,
    'metrics': {
        'regression': 'mean_absolute_percentage_error',
        'classification': 'f1_score'
    },
    'minimize': {
        'regression': True,
        'classification': False
    },
    'x_axis': 'benchmark_time',
    'size': 'predict_time',
    'ratio_max_range': 1.2,
    'colors': {
        'autogluon': '#7e8d50', 'autokeras': '#ac4142',
        'autopytorch': '#6c99ba', 'autosklearn': '#e5b566', 'flaml': '#a16a94',
        'h2o': '#7dd5cf', 'honcaml': '#505050', 'lightautoml': '#cc7833',
        'tpot': '#151515'},
    'markers': {
        'autogluon': '*', 'autokeras': 'o', 'autopytorch': 'v',
        'autosklearn': '^', 'flaml': 's', 'h2o': 'p', 'honcaml': 'P',
        'lightautoml': 'D', 'tpot': '<'}
}
