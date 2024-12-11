"""
Extract, transform, and load the air quality dataset.

This is a very short pipeline, offering just a single public function.
We put it in its own module for rapid import with few bulky deps,
as otherwise the ydata package would pull in too many deps.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

TEMP = Path("/tmp")


def get_air_quality_dataset(*, verbose: bool = False) -> pd.DataFrame:
    air_quality = fetch_ucirepo(id=360)  # Italian pollution measurements
    assert air_quality.data
    assert air_quality.metadata
    assert air_quality.metadata["name"] == "Air Quality"
    assert air_quality.metadata["characteristics"] == ["Multivariate", "Time-Series"]
    if verbose:
        vi = air_quality.metadata["additional_info"]["variable_info"]
        print(vi, "\n")

    x = air_quality.data.features
    assert isinstance(x, pd.DataFrame)
    assert air_quality.data.targets is None

    x["stamp"] = pd.to_datetime(x.Date + " " + x.Time, format="%m/%d/%Y %H:%M:%S")
    x["Time"] = pd.to_timedelta(x.Time.values).to_numpy().astype("timedelta64[s]")
    x = x.drop(columns=["Date"])
    x = _extract_pt08_features(x)
    x = x.drop(columns=["abs_humid"])
    x = _df_neg_is_nan(x)

    for col in x.columns:
        if col != "stamp":
            x[col] = x[col].astype(float)

    return x


def _extract_pt08_features(x: pd.DataFrame) -> pd.DataFrame:
    # These are five proposed sensors being studied,
    # composed of materials such as tin oxide and tungsten oxide.
    new_names = {
        "T": "temp",
        "RH": "rel_humid",
        "AH": "abs_humid",
        "PT08.S1(CO)": "co",
        "PT08.S2(NMHC)": "nmhc",  # non-methane hydrocarbons
        "PT08.S3(NOx)": "nox",
        "PT08.S4(NO2)": "no2",
        "PT08.S5(O3)": "o3",
        "C6H6(GT)": "benzene",
    }
    x = x.rename(columns=new_names)
    x = pd.DataFrame(x[list(new_names.values())])
    assert isinstance(x, pd.DataFrame)
    return x


def _extract_reference_features(x: pd.DataFrame) -> pd.DataFrame:
    new_names = {
        "T": "temp",
        "RH": "rel_humid",
        "AH": "abs_humid",
        "CO(GT)": "co",
        "NMHC(GT)": "nmhc",
        "NOx(GT)": "nox",
        "NO2(GT)": "no2",
        "C6H6(GT)": "benzene",
    }
    x = x.rename(columns=new_names)
    return pd.DataFrame(x[list(new_names.values())])


def _series_neg_is_nan(x: "pd.Series[float]", sentinel: int = -200) -> "pd.Series[float]":
    """In this dataset, it turns out -200 is used to represent missing values."""

    def to_nan(v: float) -> float:
        return v if v != sentinel else np.nan

    return pd.Series(x.apply(to_nan))


def _df_neg_is_nan(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(x.apply(_series_neg_is_nan))
