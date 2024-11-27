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
    x = _extract_reference_features(x).drop(columns=['nmhc',"abs_humid"])

    x.info()
    print(x.describe())
    print(x.corr())

    sns.pairplot(x)
    plt.show()

    # _plot_correlation(x)


def _plot_correlation(x: pd.DataFrame) -> None:
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


if __name__ == "__main__":
    main()
