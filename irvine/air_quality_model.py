#! /usr/bin/env python
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import Tensor, nn, optim
from tqdm import tqdm

from irvine.air_quality_etl import get_air_quality_dataset
from irvine.tuning import load_or_search_for_elastic_hyperparams, load_or_search_for_svr_hyperparams


@beartype
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int = 200) -> None:
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
    model: ModelType,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> dict[str, float]:
    assert isinstance(model, LSTM)
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
            "r2": float(r2_score(y_test, y_pred.numpy())),
        }


def main() -> None:
    df = get_air_quality_dataset()
    df = df.drop(columns=["stamp"])
    df = df.dropna(subset=["benzene"])
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


@beartype
def create_models(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> dict[
    str,
    tuple[
        Callable[
            [
                ModelType,
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
            ],
            dict[str, float],
        ],
        ElasticNet
        | HistGradientBoostingRegressor
        | KNeighborsRegressor
        | LSTM
        | LinearRegression
        | RandomForestRegressor
        | SVR,
    ],
]:
    tesk = train_evaluate_sklearn_model
    return {
        "ElasticNet": (tesk, load_or_search_for_elastic_hyperparams(x_train, y_train)),
        "HistGradientBoostingRegressor": (tesk, HistGradientBoostingRegressor()),
        "K-Nearest Neighbors": (tesk, KNeighborsRegressor()),
        "LSTM": (train_evaluate_lstm_model, LSTM(input_size=x_train.shape[1])),
        "LinearRegression": (tesk, LinearRegression()),
        "RandomForestRegressor": (tesk, RandomForestRegressor()),
        "SVR-RBF": (
            tesk,
            load_or_search_for_svr_hyperparams(
                x_train,
                y_train,
            ),
        ),
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
