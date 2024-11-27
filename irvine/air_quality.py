#! /usr/bin/env python

import pandas as pd
from ucimlrepo import fetch_ucirepo


def main() -> None:
    air_quality = fetch_ucirepo(id=360)  # Italian pollution measurements
    assert air_quality.data
    assert air_quality.metadata
    assert air_quality.metadata["name"] == "Air Quality"
    assert air_quality.metadata["characteristics"] == ["Multivariate", "Time-Series"]
    vi = air_quality.metadata["additional_info"]["variable_info"]
    print(vi, "\n")

    # data (as pandas dataframes)
    x = air_quality.data.features
    assert air_quality.data.targets is None

    print(x.info())
    print(x.describe())

    x["stamp"] = pd.to_datetime(x["Date"] + " " + x["Time"], format="%m/%d/%Y %H:%M:%S")
    x = x.drop(columns=["Date", "Time"])

    print(x.corr())


if __name__ == "__main__":
    main()
