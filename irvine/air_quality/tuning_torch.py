import numpy as np
import optuna
from numpy.typing import NDArray

from irvine.air_quality.aq_model import LSTM, train_evaluate_lstm_model


def lstm_error_objective(
    trial: optuna.Trial,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> float:
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)  # learning rate
    num_epochs = trial.suggest_int("num_epochs", 50, 200)
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 100, 300)

    model = LSTM(input_size=x_train.shape[1], hidden_layer_size=hidden_layer_size)

    result = train_evaluate_lstm_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=num_epochs,
        lr=lr,
        hidden_layer_size=hidden_layer_size,
    )

    return result["rmse"]
