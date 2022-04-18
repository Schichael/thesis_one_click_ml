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
