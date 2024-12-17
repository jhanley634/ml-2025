#! /usr/bin/env python

from collections.abc import Callable

import numpy as np
import pandas as pd
from beartype import beartype
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from irvine.air_quality.aq_etl import find_derivatives, get_air_quality_dataset
from irvine.air_quality.lstm_model import (
    LSTM,
    ModelType,
    randomly_sample_lstm_hyperparams_old,
    train_evaluate_lstm_model,
)
from irvine.air_quality.tuning_sklearn import (
    load_or_search_for_elastic_hyperparams,
    load_or_search_for_svr_hyperparams,
)


def train_evaluate_sklearn_model(
    model: ModelType,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> dict[str, float]:

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return {
        "rmse": float(mean_squared_error(y_test, y_pred)),
        "r2": _score(model, x_test, y_test),
    }


def _score(
    model: ModelType,
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> float:
    score = model.score(x_test, y_test)
    assert isinstance(score, float)
    return score


def main() -> None:
    df = find_derivatives(get_air_quality_dataset())

    df = df.drop(columns=["stamp"])
    holdout_split = 1800
    holdout = df.tail(holdout_split)
    df = df.head(len(df) - holdout_split)
    assert len(holdout) == holdout_split
    assert len(df) == 7190

    y = df["benzene"]
    x = df.drop(columns=["benzene"])

    x_train, x_test, y_train, y_test = _train_test_split(x, y.to_numpy())

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    assert isinstance(x_test, np.ndarray)
    x_test = x_test.astype(np.float64)

    models = create_models(x_train, y_train)

    results = {}
    for name, (train_and_evaluate_model, model) in models.items():
        print(name)
        results[name] = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)

    report(results)


@beartype
def _train_test_split(
    x: pd.DataFrame,
    y: NDArray[np.float64],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """This is simply a type safe wrapper of the familiar sklearn function."""
    kwargs = {"test_size": test_size, "random_state": seed}

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, **kwargs)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)

    return (
        x_train.to_numpy(),
        x_test.to_numpy(),
        pd.Series(y_train).to_numpy(),
        pd.Series(y_test).to_numpy(),
    )


ModelTrainer = Callable[
    [ModelType, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    dict[str, float],
]


@beartype
def create_models(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> dict[
    str,
    tuple[
        ModelTrainer,
        ElasticNet
        | HistGradientBoostingRegressor
        | KNeighborsRegressor
        | LSTM
        | LinearRegression
        | RandomForestRegressor
        | SVR,
    ],
]:
    mid = len(x_train) // 2
    train = (x_train[:mid, :], y_train[:mid])
    test = (x_train[mid:, :], y_train[mid:])
    tesk = train_evaluate_sklearn_model
    return {
        "ElasticNet": (tesk, load_or_search_for_elastic_hyperparams(x_train, y_train)),
        "HistGradientBoostingRegressor": (tesk, HistGradientBoostingRegressor()),
        "K-Nearest Neighbors": (tesk, KNeighborsRegressor()),
        "LSTM": (train_evaluate_lstm_model, randomly_sample_lstm_hyperparams_old(*train, *test)),
        "LinearRegression": (tesk, LinearRegression()),
        "RandomForestRegressor": (tesk, RandomForestRegressor()),
        "SVR-RBF": (tesk, load_or_search_for_svr_hyperparams(x_train, y_train)),
    }


def report(results: dict[str, dict[str, float]]) -> None:
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        if metrics["r2"] is not None:
            print(f"R^2:  {metrics['r2']:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
