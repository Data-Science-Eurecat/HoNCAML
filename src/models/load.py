from typing import Dict
import joblib
import os


def save_model(model: object, settings: Dict) -> None:
    filepath = os.path.join(settings['path'], settings['file'])
    joblib.dump(model, filepath)
