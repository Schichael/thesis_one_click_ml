import queries

def adaptive_binning(df, column, num_bins):
    """
    1. get 1/num_bins quantile and (num_bins-1)/num_bins quantile. Everything before the first quantile is the first
    bin. Everything after the last quantile is the last bin.
    2. keep first and last bin. For the rest apply equal width binning

    Use PQL for this.
    It needs to be improved. E.g. if there are a lot of the same value(e.g. 90%), there will be the same bins for this
    value.

    :param vals:
    :param num_bins:
    :return:
    """

    queries.get_quantiles_tracetime_pql(dm_info, quantiles, time_aggregation="DAYS")

