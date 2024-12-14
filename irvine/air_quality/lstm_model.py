import random

import numpy as np
from beartype import beartype
from numpy._typing import NDArray
from torch import Tensor, nn

from irvine.air_quality.aq_models import train_evaluate_lstm_model


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


def randomly_sample_lstm_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    *,
    n_iter: int = 10,
) -> LSTM:
    best_rmse = float("inf")
    best_model = None

    for _ in range(n_iter):
        epochs = random.randint(50, 200)
        hidden_layer_size = random.choice([50, 100, 150, 200, 250])
        lr = 10 ** random.uniform(-5, -2)
        assert 1e-5 <= lr < 1e-2

        model = LSTM(input_size=x_train.shape[1], hidden_layer_size=hidden_layer_size)

        metrics = train_evaluate_lstm_model(model, x_train, y_train, x_test, y_test, epochs, lr)

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model = model

    assert best_model
    return best_model
