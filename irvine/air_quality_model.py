#! /usr/bin/env python
import json
from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from numpy.typing import NDArray
from scipy.stats import loguniform, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import Tensor, nn, optim
from tqdm import tqdm

from irvine.air_quality_etl import TEMP, get_air_quality_dataset


@beartype
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int = 50) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        ret = self.fc(lstm_out[:, -1, :])  # Get the last LSTM output
        assert isinstance(ret, Tensor)
        return ret


ModelType = TypeVar(
    "ModelType",
    ElasticNet,
    HistGradientBoostingRegressor,
    KNeighborsRegressor,
    LSTM,
    LinearRegression,
    RandomForestRegressor,
    # RandomizedSearchCV,
    SVR,
)


@beartype
def train_evaluate_sklearn_model(
    model: ModelType,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> dict[str, float]:
    if isinstance(model, LSTM):
        return train_evaluate_lstm_model(model, x_train, y_train, x_test, y_test)

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


def train_evaluate_lstm_model(
    model: LSTM,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: Iterable[float],
) -> dict[str, float]:
    epochs: int = 200
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.04)
    x_train_tensor = Tensor(x_train).unsqueeze(1)
    x_test_tensor = Tensor(x_test).unsqueeze(1)
    y_train_tensor = Tensor(y_train).unsqueeze(-1)  # Adds a dimension to make it [7192, 1]

    for name, param in model.lstm.named_parameters():
        if "weight" in name:
            torch.nn.init.xavier_uniform_(param)
        elif "bias" in name:
            torch.nn.init.zeros_(param)

    for _ in tqdm(range(epochs), leave=False):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor)
        return {
            "rmse": float(mean_squared_error(y_test, y_pred)),
            "r2": 0.0,
        }


def main() -> None:
    df = get_air_quality_dataset().dropna(subset=["benzene"])
    holdout_split = 1800
    holdout = df.tail(holdout_split)
    df = df.head(len(df) - holdout_split)
    assert len(holdout) == holdout_split
    assert len(df) == 7191

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
    for name, model in models.items():
        print(name)
        results[name] = train_evaluate_sklearn_model(model, x_train, y_train, x_test, y_test)

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, **kwargs)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)

    return (
        x_train.to_numpy(),
        x_test.to_numpy(),
        pd.Series(y_train).to_numpy(),
        pd.Series(y_test).to_numpy(),
    )


@beartype
def create_models(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> dict[str, ModelType]:
    models = {
        "ElasticNet": ElasticNet(),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        # "LSTM": LSTM(input_size=x_train.shape[1]),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR-RBF": load_or_search_for_svr_hyperparams(
            x_train,
            y_train,
        ),
    }
    types = (
        ElasticNet
        | HistGradientBoostingRegressor
        | KNeighborsRegressor
        | LinearRegression
        | RandomForestRegressor
        | SVR
    )
    assert types
    # for model in models.values():
    #     assert isinstance(model, types)
    return models


PARAM_CACHE = TEMP / "svr_params.json"


def load_or_search_for_svr_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> SVR:
    if not PARAM_CACHE.exists():

        svr_search = search_for_svr_hyperparams()
        svr_search.fit(x_train, y_train)

        with PARAM_CACHE.open("w") as fout:
            json.dump({k: [v] for k, v in svr_search.best_params_.items()}, fout)

    with PARAM_CACHE.open() as fin:
        best_params = json.load(fin)
        return SVR(**best_params)


def search_for_svr_hyperparams() -> RandomizedSearchCV:
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


def report(results: dict[str, dict[str, float]]) -> None:
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        if metrics["r2"] is not None:
            print(f"R^2:  {metrics['r2']:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
