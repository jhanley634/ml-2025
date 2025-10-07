"""
Extract, transform, and load the air quality dataset.

This is a very short pipeline, offering just a few public functions.
We put it in its own module for rapid import with few bulky deps,
as otherwise the ydata package would pull in too many deps.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

TEMP = Path("/tmp")

NANO_PER_SEC = int(1e9)


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
    x["hour"] = 24 * x.stamp.dt.day_of_week + x.stamp.dt.hour  # 168 hourly buckets
    x["stamp"] = x.stamp.astype(int) // NANO_PER_SEC  # seconds since 1970
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
        "stamp": "stamp",
        "hour": "hour",
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


def synthesize_features(df: pd.DataFrame) -> pd.DataFrame:
    return _find_derivatives(_weekend(df))


def _weekend(df: pd.DataFrame) -> pd.DataFrame:
    """0 / 1 / 2 for weekday / Saturday / Sunday.

    Saturday shows low pollution, but is higher than Sunday.
    Weekdays OTOH are relatively indistinguishable.
    """
    sat, sun = 5, 6
    weekend_map = dict.fromkeys(range(5), 0)
    weekend_map.update({sat: 1, sun: 2})
    df["weekend"] = pd.to_datetime(df.stamp, unit="s").dt.day_of_week.map(weekend_map)
    return df


def _find_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["benzene"])  # 9357 --> 8991 rows
    n = len(df)
    df = df.dropna(subset=["o3"])
    assert n == len(df)

    df["dt"] = df.stamp.diff()
    for col in ["benzene", "co", "nmhc", "nox", "no2", "o3", "temp"]:
        df[f"{col}_deriv"] = df[col].diff() / df.dt
    return df.dropna(subset=["benzene_deriv"])  # discard first row


@beartype
def arr(x: pd.DataFrame | pd.Series) -> NDArray[np.float64]:
    return np.array(x.to_numpy(), dtype=np.float64)


@beartype
def aq_train_test_split(
    x: pd.DataFrame,
    y: NDArray[np.float64],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """This is simply a type safe wrapper of the familiar sklearn function."""
    kwargs = {"test_size": test_size, "random_state": seed}

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, **kwargs)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)

    return (
        arr(x_train),
        arr(x_test),
        arr(pd.Series(y_train)),
        arr(pd.Series(y_test)),
    )
