#! /usr/bin/env python
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble._forest import ForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import Tensor, float32, nn, optim, tensor

from irvine.air_quality_eda import get_air_quality_dataset


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


def train_evaluate_sklearn_model(
    model: ForestRegressor | HistGradientBoostingRegressor | LinearRegression | SVR,
    x_train: Tensor,
    y_train: Iterable[float],
    x_test: NDArray[np.float64],
    y_test: Iterable[float],
) -> dict[str, float]:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {
        "rmse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(model.score(x_test, y_test)),
    }


def train_evaluate_lstm(
    x_train: Tensor,
    y_train: NDArray[np.float64],
    x_test: Tensor,
    y_test: Iterable[float],
    epochs: int = 10,
) -> dict[str, float]:
    model = LSTM(input_size=x_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    y_train_tensor = Tensor(y_train).unsqueeze(-1)  # Adds a dimension to make it [7192, 1]

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        return {"rmse": mean_squared_error(y_test, y_pred), "r2": 0.0}


def main() -> None:
    df = get_air_quality_dataset().dropna(subset=["benzene"])
    y = df["benzene"]
    x = df.drop(columns=["benzene"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    assert isinstance(y_train, pd.Series)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    assert isinstance(x_test_scaled, np.ndarray)

    models = {
        "HistGradientBoosting": HistGradientBoostingRegressor(),
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR(),
        "LinearRegression": LinearRegression(),
    }

    results = {}
    for name, model in models.items():
        results[name] = train_evaluate_sklearn_model(
            model,
            x_train_scaled,
            y_train,
            x_test_scaled.astype(np.float64),
            y_test,
        )

    x_train_lstm = tensor(x_train_scaled, dtype=float32).unsqueeze(1)
    x_test_lstm = tensor(x_test_scaled, dtype=float32).unsqueeze(1)

    results["LSTM"] = train_evaluate_lstm(x_train_lstm, y_train.to_numpy(), x_test_lstm, y_test)

    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"RMSE: {metrics['rmse']}")
        if metrics["r2"] is not None:
            print(f"R^2: {metrics['r2']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
