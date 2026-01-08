import unittest

import numpy as np
import pandas as pd
from pandas import Series

# from https://stackoverflow.com/questions/79842390/if-not-working-with-pd-notna


class BoolTypeCheckTest(unittest.TestCase):

    def test_foo(self, *, verbose: bool = False) -> None:
        df = pd.DataFrame(
            [
                {"a": 1, "b": True},
                {"a": 2, "b": False},
                {"a": 3, "b": True},
            ],
        )
        assert type(df.b) is Series
        assert type(df.b[0]) is np.bool

        if verbose:
            print(df.dtypes)
            for row in df.itertuples():
                if row.b:
                    print(row.a)

            print(df.b.any())
            print(df.b.all())
