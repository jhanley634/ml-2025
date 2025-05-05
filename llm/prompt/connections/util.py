import re
from pathlib import Path

import pandas as pd

temp = Path("/tmp") / "connections_result.csv"


def as_df(markdown_table: str) -> pd.DataFrame:
    text = "CATEGORY|WORDS\n" + canonicalize(markdown_table.lstrip())
    temp.write_text(text)
    return pd.read_csv(temp, sep="|", engine="python")


def canonicalize(markdown_table: str) -> str:
    r"""Makes a markdown formatted table look more like a CSV.

    >>> canonicalize("| a b   | c d e f   |\n|-----|-------|")
    'a b|c d e f\n----|----'
    """
    tbl = markdown_table
    for regex, repl in [
        (r" +", " "),  # no repeated blanks
        (r" \| ", "|"),  # lstrip() the 2nd column
        (r"----+", "----"),  # always 4 dashes
        (r"^\| *| *\|$", ""),  # '|x|y|' --> 'x|y'
    ]:
        tbl = re.sub(regex, repl, tbl, flags=re.MULTILINE)
    return tbl


def validate(df: pd.DataFrame) -> None:
    for row in df.itertuples():
        category = f"{row.CATEGORY}"
        words = f"{row.WORDS}"
        assert category == category.upper(), row
        assert words == words.upper(), row
        commas = words.count(",")
        if commas != 3:
            print(commas, row)
