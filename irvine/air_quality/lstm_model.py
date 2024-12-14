import random

import numpy as np
import torch
from beartype import beartype
from numpy._typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from torch import Tensor, nn

from irvine.air_quality.aq_models import train_evaluate_lstm_model


class SklearnLSTMWrapper(BaseEstimator, ClassifierMixin):  # type: ignore [misc]
    def __init__(
        self,
        input_size: int,
        hidden_layer_size: int = 200,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.lr = lr
        self.model = LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size)
        self.rmse = float("inf")

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> "SklearnLSTMWrapper":
        self.model = LSTM(input_size=self.input_size, hidden_layer_size=self.hidden_layer_size)

        metrics = train_evaluate_lstm_model(self.model, x, y, x, y, self.epochs, self.lr)
        self.rmse = metrics["rmse"]
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        with torch.no_grad():
            assert self.model
            self.model.eval()
            inputs = torch.tensor(x, dtype=torch.float32)
            output = self.model(inputs)
            return np.ndarray(output.numpy())

    def score(
        self,
        X: NDArray[np.float64],  # noqa: N803
        y: NDArray[np.float64],
        sample_weight: float | None = None,
    ) -> float:
        assert sample_weight is None
        predictions = self.predict(X)
        return float(mean_squared_error(y, predictions))


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


def randomly_sample_lstm_hyperparams_unused(
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


def randomly_sample_lstm_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    *,
    n_iter: int = 10,
) -> LSTM:

    assert x_test != y_test

    param_dist = {
        "hidden_layer_size": [50, 100, 150, 200, 250],
        "epochs": [50, 100, 150, 200],
        "lr": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    }

    # Wrap the model in SklearnLSTMWrapper
    model = SklearnLSTMWrapper(input_size=x_train.shape[1])

    # Initialize the RandomizedSearchCV with the desired search space
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",  # Use RMSE as the scoring metric
        cv=3,  # 3-fold cross-validation
        verbose=1,
        random_state=42,
    )

    # Perform the hyperparameter search
    random_search.fit(x_train, y_train)

    # Print the best hyperparameters and RMSE
    print(f"Best hyperparameters found: {random_search.best_params_}")
    print(f"Best RMSE: {-random_search.best_score_}")  # Negate to get the positive RMSE

    # Get the best model from the search
    best_model = random_search.best_estimator_

    # Return the best model (which has been trained with the best hyperparameters)
    assert isinstance(best_model.model, LSTM)
    return best_model.model
