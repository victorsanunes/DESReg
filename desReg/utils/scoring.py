import numpy as np
from collections import Counter

from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _extract_total_and_distinct(c: Counter):
    # Prefer explicit keys if present (DES provides them)
    if 'TotalRegressors' in c:
        total = c['TotalRegressors']
    else:
        total = int(sum(v for k, v in c.items() if not k.endswith('Regressors')))

    if 'DistinctRegressors' in c:
        distinct = c['DistinctRegressors']
    else:
        distinct = int(len([k for k in c.keys() if not k.endswith('Regressors')]))

    return float(total), float(distinct)


def mean_total_regressors(estimator, X, y):
    # Custom scorer that queries ensemble sizes from estimator
    y_pred, sizes = estimator.predict(X, return_ensemble_sizes=True)
    if isinstance(sizes, list) and len(sizes) > 0:
        totals = []
        for c in sizes:
            t, _ = _extract_total_and_distinct(c)
            totals.append(t)
        return float(np.mean(totals))
    if isinstance(sizes, Counter):
        t, _ = _extract_total_and_distinct(sizes)
        return t
    return float('nan')


def mean_distinct_regressors(estimator, X, y):
    # Custom scorer that queries ensemble sizes from estimator
    y_pred, sizes = estimator.predict(X, return_ensemble_sizes=True)
    if isinstance(sizes, list) and len(sizes) > 0:
        distincts = []
        for c in sizes:
            _, d = _extract_total_and_distinct(c)
            distincts.append(d)
        return float(np.mean(distincts))
    if isinstance(sizes, Counter):
        _, d = _extract_total_and_distinct(sizes)
        return d
    return float('nan')


# Standard scorers
neg_mse = make_scorer(mean_squared_error, greater_is_better=False)
neg_rmse = make_scorer(_rmse, greater_is_better=False)
neg_mae = make_scorer(mean_absolute_error, greater_is_better=False)
neg_mape = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
r2 = make_scorer(r2_score, greater_is_better=True)


def get_scorers(include_ensemble_metrics: bool = True):
    scorers = {
        'neg_mse': neg_mse,
        'neg_rmse': neg_rmse,
        'neg_mae': neg_mae,
        'neg_mape': neg_mape,
        'r2': r2,
    }
    if include_ensemble_metrics:
        scorers.update(
            {
                'mean_total_regressors': mean_total_regressors,
                'mean_distinct_regressors': mean_distinct_regressors,
            }
        )
    return scorers


__all__ = [
    'neg_mse',
    'neg_rmse',
    'neg_mae',
    'neg_mape',
    'r2',
    'mean_total_regressors',
    'mean_distinct_regressors',
    'get_scorers',
]


