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
    'a b | c d e f\n----|----'
    """
    tbl = re.sub(r" +", " ", markdown_table, flags=re.MULTILINE)  # no repeated blanks
    tbl = re.sub(r"----+", "----", tbl, flags=re.MULTILINE)  # always 4 dashes
    return re.sub(r"^\| *| *\|$", "", tbl, flags=re.MULTILINE)  # '|x|y|' --> 'x|y'
