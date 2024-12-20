#! /usr/bin/env python

from pprint import pprint

import autosklearn.classification
import autosklearn.regression
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from pandas.testing import assert_frame_equal

from irvine.air_quality.aq_etl import aq_train_test_split, get_air_quality_dataset


def equality_demo() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    df2 = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    assert_frame_equal(df1, df2)


def auto_learn_demo1() -> None:

    df = get_air_quality_dataset().dropna(subset=["benzene"])
    y = df["benzene"]
    x = df.drop(columns=["benzene"])
    x_train, x_test, y_train, y_test = aq_train_test_split(x, y.to_numpy())

    reg = autosklearn.regression.AutoSklearnRegressor()
    reg.fit(x_train, y_train)
    predictions = reg.predict(x_test)
    print(predictions)


def auto_learn_demo() -> None:
    x, y = sklearn.datasets.load_diabetes(return_X_y=True)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, random_state=1
    )

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder="/tmp/autosklearn_regression_example_tmp",
        memory_limit=100_000,
        n_jobs=1,
    )
    f = automl.fit(x_train, y_train, dataset_name="diabetes")
    print(f, type(f))
    print(automl.leaderboard())
    pprint(automl.show_models(), indent=4)


if __name__ == "__main__":
    auto_learn_demo()
