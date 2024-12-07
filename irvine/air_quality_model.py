#! /usr/bin/env python

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from irvine.air_quality_eda import get_air_quality_dataset


def main() -> None:
    df = get_air_quality_dataset().dropna(subset=["benzene"])
    y = df["benzene"]
    x = df.drop(columns=["benzene"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = HistGradientBoostingRegressor()
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"RMSE: {rmse}")
    print(f"R^2:  {model.score(x_test_scaled, y_test)}")
    print(f"Features: {x.columns}")
    assert model.loss == "squared_error"


if __name__ == "__main__":
    main()
