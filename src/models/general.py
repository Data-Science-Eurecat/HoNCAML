from typing import Dict, List
import numpy as np
import pandas as pd
from src.tools import utils


class Metrics:
    accuracy = 'sklearn.metrics.accuracy_score'
    f1 = 'sklearn.metrics.f1_score'


def compute_metrics(y_true: np.array, y_pred: np.array, metrics: List) -> Dict:
    results = {}
    for metric in metrics:
        results['metric'] = utils.import_library(
            getattr(Metrics, metric), {'y_true': y_true, 'y_pred': y_pred})
    return results


def aggregate_cv_results(cv_results: List[Dict]) -> Dict:
    df_results = pd.DataFrame(cv_results)
    mean_results = df_results.mean(axis=0).to_dict()
    return mean_results
