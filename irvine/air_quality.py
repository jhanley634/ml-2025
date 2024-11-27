#! /usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    x["Time"] = pd.to_timedelta(x["Time"])
    x = x.drop(columns=["Date"])

    x.info()
    print(x.describe())
    print(x.corr())
    # plot_correlation(x)


def plot_correlation(x: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(x.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation")
    plt.show()


if __name__ == "__main__":
    main()
