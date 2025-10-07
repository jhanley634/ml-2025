#! /usr/bin/env python

# from https://stackoverflow.com/questions/79784978/webpage-table-as-a-dataframe

from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get(url: str) -> StringIO:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return StringIO(resp.text)


url = "https://www.cbp.gov/newsroom/stats/southwest-land-border-encounters"


def main() -> None:
    soup = BeautifulSoup(get(url), "lxml")
    print(soup.prettify())
    # pd.read_html(get(url))


if __name__ == "__main__":
    main()
