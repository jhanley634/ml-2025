import unittest

import numpy as np
import pandas as pd
from pandas import Series

# from https://stackoverflow.com/questions/79842390/conditionnal-if-use-not-working-with-pd-notna


class BoolTypeCheckTest(unittest.TestCase):

    def test_foo(self) -> None:
        df = pd.DataFrame(
            [
                {"a": 1, "b": True},
                {"a": 2, "b": False},
                {"a": 3, "b": True},
            ],
        )
        assert type(df.b) is Series
        assert type(df.b[0]) is np.bool
        print(df.dtypes)

        for row in df.itertuples():
            if row.b:
                print(row.a)

        print(df.b.any())
        print(df.b.all())

        # if df.b:  print("foo")
