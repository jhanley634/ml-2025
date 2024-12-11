import json

import numpy as np
from numpy.typing import NDArray
from scipy.stats import loguniform, uniform
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from irvine.air_quality_etl import TEMP

ELASTIC_CACHE = TEMP / "elasticnet_params_cache.json"
SVR_CACHE = TEMP / "svr_params.json"


def load_or_search_for_elastic_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> ElasticNet:
    if not ELASTIC_CACHE.exists():
        elastic_net_search = _search_for_elastic_hyperparams()
        elastic_net_search.fit(x_train, y_train)

        with ELASTIC_CACHE.open("w") as fout:
            json.dump(elastic_net_search.best_params_, fout)
            fout.write("\n")

    with ELASTIC_CACHE.open() as fin:
        best_params = json.load(fin)

    return ElasticNet(**best_params)


def load_or_search_for_svr_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> SVR:
    if not SVR_CACHE.exists():

        svr_search = _search_for_svr_hyperparams()
        svr_search.fit(x_train, y_train)

        with SVR_CACHE.open("w") as fout:
            json.dump(svr_search.best_params_, fout)
            fout.write("\n")

    with SVR_CACHE.open() as fin:
        best_params = json.load(fin)
        return SVR(**best_params)


def _search_for_elastic_hyperparams() -> RandomizedSearchCV:
    elastic_param_grid = {
        "alpha": loguniform(1e-3, 1.0),
        "l1_ratio": uniform(0.0, 1.0),
        "max_iter": [22_000, 26_000, 30_000, 34_000, 38_000],
        "tol": [1e-3],
    }
    return RandomizedSearchCV(
        ElasticNet(),
        elastic_param_grid,
        n_iter=30,
        cv=4,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )


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
