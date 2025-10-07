#! /usr/bin/env python

# from https://stackoverflow.com/questions/79784978/webpage-table-as-a-dataframe

import pandas as pd

if __name__ == "__main__":
    pd.read_html("https://www.cbp.gov/newsroom/stats/southwest-land-border-encounters")
