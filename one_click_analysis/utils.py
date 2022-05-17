import datetime
from typing import Any
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from pycelonis import get_celonis


def join_dfs(dfs: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
    """Perform a Left outer join on two DataFrame. Only the key of the first
    DataFrame is kept

    :param dfs: list of at least two DataFrames
    :param keys: columns to join on. But be of same length as dfs
    :return: joined DataFrame
    """
    df_result = None
    for i in range(0, len(dfs)):
        if dfs[i] is not None:
            df_result = dfs[i]
            break

    for i in range(1, len(dfs)):
        if dfs[i] is None:
            continue
        # Rmove common columns from one of those
        common_columns = np.intersect1d(df_result.columns, dfs[i].columns).tolist()
        if keys[i] in common_columns:
            common_columns.remove(keys[i])
        dfs[i] = dfs[i].drop(common_columns, axis=1)
        df_result = pd.merge(
            df_result, dfs[i], how="left", left_on=keys[0], right_on=keys[i]
        )

        # Drop right key if it's different from the left key
        if keys[0] != keys[i]:
            df_result.drop(keys[i], axis=1)
    return df_result


def get_dm(datamodel_name: str, celonis_login: Optional[dict] = None):
    """Get datamodel from Celonis

    :param datamodel_name: name or id of datamodel
    :param celonis_login: login data for Celonis
    :return:
    """
    if celonis_login is None:
        celonis = get_celonis()
    else:
        celonis = get_celonis(**celonis_login)
    dm = celonis.datamodels.find(datamodel_name)
    return dm


def make_list(var: Any) -> List:
    """Wraps varibale var into a list if val is not a list itself.

    :param var: a variable
    :return: var if var is of type list or empty list if var=None, else [var]
    """
    if var is None:
        return []
    if isinstance(var, List):
        return var
    else:
        return [var]


def convert_date_to_str(d: datetime.date) -> str:
    """Convert a datetime.date object to a date string

    :param d: input date with a year, month and day
    :return: string of date of form yyyy-mm-dd
    """
    year = str(d.year)
    month = str(d.month) if d.month >= 10 else "0" + str(d.month)
    day = str(d.day) if d.day >= 10 else "0" + str(d.day)
    date_str = f"{year}-{month}-{day}"
    return date_str


def get_aggregation_df_name(agg: str):
    """Generate the name of the aggregation to display to the user from the
    aggregation String that is used for a PQL query

    :param agg: original aggregation name as used for
    :return: aggregation string to display
    """
    if agg == "MIN":
        return "minimum"
    elif agg == "MAX":
        return "maximum"
    elif agg == "AVG":
        return "mean"
    elif agg == "MEDIAN":
        return "median"
    elif agg == "FIRST":
        return "first"
    elif agg == "LAST":
        return "last"


def list_intersection(list_a: List[Any], list_b: List[Any]):
    """Intersection of two lists. Result will be unordered.

    :param list_a: First list
    :param list_b: Second list
    :return: List of the intersection of list_a and list_b
    """
    return list(set(list_a) & set(list_b))
