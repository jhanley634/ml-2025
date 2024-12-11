import json

import numpy as np
from numpy._typing import NDArray
from scipy.stats import loguniform, uniform
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from irvine.air_quality_model import PARAM_CACHE


def load_or_search_for_elastic_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> ElasticNet:
    pass


def load_or_search_for_svr_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> SVR:
    if not PARAM_CACHE.exists():

        svr_search = _search_for_svr_hyperparams()
        svr_search.fit(x_train, y_train)

        with PARAM_CACHE.open("w") as fout:
            json.dump(svr_search.best_params_, fout)

    with PARAM_CACHE.open() as fin:
        best_params = json.load(fin)
        return SVR(**best_params)


def _search_for_elastic_hyperparams() -> RandomizedSearchCV:
    pass


def _search_for_svr_hyperparams() -> RandomizedSearchCV:
    svr_param_grid = {
        "C": loguniform(1e-2, 1e3),
        "epsilon": uniform(0.01, 0.5),
        "kernel": ["linear", "poly", "rbf"],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4],
        "coef0": uniform(0, 10),
    }
    return RandomizedSearchCV(
        SVR(kernel="rbf"),
        svr_param_grid,
        n_iter=20,
        cv=4,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
