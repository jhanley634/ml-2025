#! /usr/bin/env python
"""
Exploratory data analysis for the air quality dataset.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport

from irvine.air_quality.aq_etl import TEMP, get_air_quality_dataset


def main() -> None:
    x = get_air_quality_dataset()
    x.info()
    print(x.describe())
    print(x.corr())
    print(x)

    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    x.to_csv(TEMP / "data.csv", index=False)

    _plot_correlation(x)

    pr = ProfileReport(x)
    pr.to_file(TEMP / "air_quality_report.html")


def _plot_correlation(x: pd.DataFrame) -> None:
    sns.pairplot(x)

    plt.figure(figsize=(10, 8))
    sns.heatmap(x.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation")
    plt.show()


def _z_scale(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.mean()) / x.std()


if __name__ == "__main__":
    main()
