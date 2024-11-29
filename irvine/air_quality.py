#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def main() -> None:
    air_quality = fetch_ucirepo(id=360)  # Italian pollution measurements
    assert air_quality.data
    assert air_quality.metadata
    assert air_quality.metadata["name"] == "Air Quality"
    assert air_quality.metadata["characteristics"] == ["Multivariate", "Time-Series"]
    vi = air_quality.metadata["additional_info"]["variable_info"]
    print(vi, "\n")

    x = air_quality.data.features
    assert isinstance(x, pd.DataFrame)
    assert air_quality.data.targets is None

    x["stamp"] = pd.to_datetime(x["Date"] + " " + x["Time"], format="%m/%d/%Y %H:%M:%S")
    x["Time"] = pd.to_timedelta(x["Time"]).dt.total_seconds()
    x = x.drop(columns=["Date"])
    x = _extract_reference_features(x)
    x = x.drop(columns=["abs_humid"])
    x = _df_neg_is_nan(x)

    for col in x.columns:
        if col != "stamp":
            x[col] = x[col].astype(float)

    x.info()
    print(x.describe())
    print(x.corr())
    print()
    print(x)

    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    x.to_csv("/tmp/data.csv", index=False)

    _plot_correlation(x)


def _plot_correlation(x: pd.DataFrame) -> None:
    sns.pairplot(x)

    plt.figure(figsize=(10, 8))
    sns.heatmap(x.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation")
    plt.show()


def _extract_reference_features(x: pd.DataFrame) -> pd.DataFrame:
    new_names = {
        "CO(GT)": "co",
        "NMHC(GT)": "nmhc",  # non-methane hydrocarbons
        "C6H6(GT)": "benzene",
        "NOx(GT)": "nox",
        "NO2(GT)": "no2",
        "T": "temp",
        "RH": "rel_humid",
        "AH": "abs_humid",
    }
    x = x.rename(columns=new_names)
    return x[list(new_names.values())]


def _series_neg_is_nan(x: "pd.Series[float]", sentinel: int = -200) -> "pd.Series[float]":
    """In this dataset, it turns out -200 is used to represent missing values."""

    def to_nan(v: float) -> float:
        return v if v != sentinel else np.nan

    return x.apply(to_nan)


def _df_neg_is_nan(x: pd.DataFrame) -> pd.DataFrame:
    return x.apply(_series_neg_is_nan)


def _z_scale(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.mean()) / x.std()


if __name__ == "__main__":
    main()
