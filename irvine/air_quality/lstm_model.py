import random
from typing import TypeVar

import numpy as np
import torch
from beartype import beartype
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from torch import Tensor, nn, optim
from tqdm import tqdm


@beartype
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

            # Ensure the input has 3 dimensions (batch_size, seq_length, input_size)
            # Assuming you are working with a sequence length of 1 (univariate time series)
            assert inputs.ndimension() == 2, f"need a 2D tensor, {inputs.ndimension()=}"
            inputs = inputs.unsqueeze(1)  # .shape is now [batch_size, 1, input_size]
            assert inputs.ndimension() == 3, f"{inputs.ndimension()=}"

            output = self.model(inputs)
            return np.array(output.numpy())

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
    def __init__(self, input_size: int, hidden_layer_size: int = 100) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        assert lstm_out.ndimension() == 3, f"need a 3D tensor, {lstm_out.ndimension()=}"
        # assert lstm_out.shape[1:] == (1, self.input_size)
        last_hidden = self.fc(lstm_out[:, -1, :])
        assert isinstance(last_hidden, Tensor)
        assert last_hidden.shape[1:] == (1,)
        return last_hidden


@beartype
def randomly_sample_lstm_hyperparams_old(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    *,
    n_iter: int = 10,
) -> LSTM:
    assert 2 == x_train.ndim == x_test.ndim
    assert 1 == y_train.ndim == y_test.ndim

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


@beartype
def randomly_sample_lstm_hyperparams(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    *,
    n_iter: int = 10,
) -> LSTM:
    assert len(x_test) > 0
    assert len(y_test) > 0

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

ModelType = TypeVar("ModelType")


@beartype
def train_evaluate_lstm_model[ModelType](  # noqa: PLR0913
    model: ModelType,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    epochs: int = 100,
    learning_rate: float = 0.04,
) -> dict[str, float]:
    assert isinstance(model, LSTM)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    x_train_tensor = Tensor(x_train).unsqueeze(1)  # shape [batch_size, seq_length=1, input_size]
    x_test_tensor = Tensor(x_test).unsqueeze(1)  # shape [batch_size, seq_length=1, input_size]
    y_train_tensor = Tensor(y_train).unsqueeze(-1)  # shape [batch_size, 1] for regression tasks

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
