#! /usr/bin/env python

import autosklearn.classification
import pandas as pd
from pandas.testing import assert_frame_equal

from irvine.air_quality.aq_etl import get_air_quality_dataset
from irvine.air_quality.aq_models import aq_train_test_split


def equality_demo() -> None:
    df1 = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    df2 = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    assert_frame_equal(df1, df2)


def auto_learn_demo() -> None:

    df = get_air_quality_dataset()
    y = df["benzene"]
    x = df.drop(columns=["benzene"])
    x_train, x_test, y_train, y_test = aq_train_test_split(x, y.to_numpy())

    cls = autosklearn.classification.AutoSklearnClassifier()
    cls.fit(x_train, y_train)
    predictions = cls.predict(x_test)
    print(predictions)


if __name__ == "__main__":
    auto_learn_demo()
