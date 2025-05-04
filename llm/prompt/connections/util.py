import re
from pathlib import Path

import pandas as pd

temp = Path("/tmp") / "connections_result.csv"


def as_df(markdown_table: str) -> pd.DataFrame:
    text = "category|words\n" + _strip_leading_and_trailing_pipes(markdown_table.lstrip())
    temp.write_text(text)
    return pd.read_csv(temp, sep="|", engine="python")


def _strip_leading_and_trailing_pipes(markdown_table: str) -> str:
    r"""Makes a markdown formatted table look more like a CSV.

    >>> _strip_leading_and_trailing_pipes("| a b | c d e f |\n|---|---|")
    'a b|c d e f\n---|---'
    """
    tbl = re.sub(r" *\| *", "|", markdown_table, flags=re.MULTILINE)  # 'x | y' --> 'x|y'
    return re.sub(r"^\||\|$", "", tbl, flags=re.MULTILINE)
