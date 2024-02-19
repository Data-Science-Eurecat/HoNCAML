default_data_step = {
    "transform": {
        "encoding": {
            "OHE": True,
            "max_values": 100
        },
        "normalize": {
            "features": {
                "module": "sklearn.preprocessing.StandardScaler"
            },
            "target": {
                "module": "sklearn.preprocessing.StandardScaler"
            }
        }
    }
}
