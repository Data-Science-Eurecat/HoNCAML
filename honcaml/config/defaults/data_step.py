default_data_step = {
    "extract": {
        "filepath": "data/raw/dataset.csv",
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
    },
    "load": {"filepath": "data/processed/dataset.csv"}
}
