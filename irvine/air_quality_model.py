#! /usr/bin/env python

import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import Tensor, float32, nn, optim, tensor

from irvine.air_quality_eda import get_air_quality_dataset


class LSTM(nn.Module):  # type: ignore [misc]
    def __init__(self, input_size: int, hidden_layer_size: int = 50) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        ret = self.fc(lstm_out[:, -1, :])  # Get the last LSTM output
        assert isinstance(ret, Tensor)
        return ret


def main() -> None:
    df = get_air_quality_dataset().dropna(subset=["benzene"])
    y = df["benzene"]
    x = df.drop(columns=["benzene"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "HistGradientBoosting": HistGradientBoostingRegressor(),
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR(),
        "LinearRegression": LinearRegression(),
    }

    for model_name, model in models.items():
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        rmse = mean_squared_error(y_test, y_pred)
        print(f"Model: {model_name}")
        print(f"RMSE: {rmse}")
        print(f"R^2:  {model.score(x_test_scaled, y_test)}")
        print(f"Features: {x.columns}")
        print("-" * 40)

    x_train_lstm = tensor(x_train_scaled, dtype=float32).unsqueeze(
        1,
    )
    x_test_lstm = tensor(x_test_scaled, dtype=float32).unsqueeze(1)

    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    y_train_tensor = tensor(y_train.values, dtype=float32).view(-1, 1)
    y_test_tensor = tensor(y_test.values, dtype=float32).view(-1, 1)

    input_size = x_train_lstm.shape[2]  # Number of features in the input
    lstm_model = LSTM(input_size=input_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # Training the LSTM model
    num_epochs = 10
    for epoch in range(num_epochs):
        lstm_model.train()
        optimizer.zero_grad()

        assert isinstance(x_train_lstm, torch.Tensor)
        y_pred_lstm = lstm_model(x_train_lstm)

        loss = criterion(y_pred_lstm, y_train_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    lstm_model.eval()
    with torch.no_grad():
        y_pred_lstm = lstm_model(x_test_lstm)
        rmse_lstm = mean_squared_error(y_test_tensor, y_pred_lstm)
        print("Model: LSTM")
        print(f"RMSE: {rmse_lstm}")
        print("-" * 40)


if __name__ == "__main__":
    main()
