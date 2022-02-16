from pycelonis import get_celonis
from pycelonis.celonis_api.pql.pql import PQL, PQLColumn, PQLFilter
from DataModel import DataModelInfo
import numpy as np
from pycelonis.celonis_api.utils import KeyType

login = {
    "celonis_url": "academic-michael-schulten-rwth-aachen-de.eu-2.celonis.cloud",
    "api_token": "ODJkZDhjNmQtYTQ2Ny00NWRlLWJkMGYtZWJjY2FjOGVhYmQyOllBYlBkRXNHV2psZ1o1MDJacmlsRVU3KytxaDdLVHY5N1lBOHJQdTJnOXR0",
    # The following 2 lines are only necessary when connecting to CPM4.5, not for IBC:
    # "api_id": "paste_here_your_api_id",
    # "username": "paste_here_your_username",
}
celonis = get_celonis(**login)
dm = celonis.datamodels.find("P2P_Wils_Course")

dm_info = DataModelInfo(dm)


def get_case_duration_pql(dm_info, aggregation="AVG", time_aggregation="DAYS"):
    q = (
        aggregation
        + "(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO ALL_OCCURRENCE['Process End'], REMAP_TIMESTAMPS(\""
        + dm_info.activity_table_name
        + '"."'
        + dm_info.eventtime_col
        + '", '
        + time_aggregation
        + ")))"
    )
    query = PQL()
    query.add(PQLColumn(q, "average case duration"))
    df_avg_case_duration = dm_info.dm.get_data_frame(query)
    return df_avg_case_duration["average case duration"].values[0]


def get_case_duration_development_pql(
    dm_info,
    duration_aggregation="AVG",
    date_aggregation="ROUND_MONTH",
    time_aggregation="DAYS",
):
    q_date = (
        date_aggregation
        + '("'
        + dm_info.activity_table_name
        + '"."'
        + dm_info.eventtime_col
        + '")'
    )
    q_duration = (
        duration_aggregation
        + "(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO ALL_OCCURRENCE['Process End'], REMAP_TIMESTAMPS(\""
        + dm_info.activity_table_name
        + '"."'
        + dm_info.eventtime_col
        + '", '
        + time_aggregation
        + ")))"
    )
    query = PQL()
    query.add(PQLColumn(q_date, "datetime"))
    query.add(PQLColumn(q_duration, "case duration"))
    df_avg_case_duration = dm_info.dm.get_data_frame(query)
    return df_avg_case_duration


def get_quantiles_tracetime_pql(dm_info, quantiles, time_aggregation="DAYS"):
    q_quantiles = []
    for quantile in quantiles:
        q = (
            "QUANTILE(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO ALL_OCCURRENCE['Process End'], "
            'REMAP_TIMESTAMPS(" '
            + dm_info.activity_table_name
            + '"."'
            + dm_info.eventtime_col
            + '", '
            + time_aggregation
            + ")), "
            + str(quantile)
            + ")"
        )
        q_quantiles.append(q)

    query = PQL()
    for q in q_quantiles:
        query.add(PQLColumn(q[0], q[1]))

    df_quantiles = dm_info.dm.get_data_frame(query)
    return df_quantiles


def get_num_cases_with_durations(dm_info, durations, time_aggregation="DAYS"):
    """Get the number of cases with the durations.

    :param dm_info:
    :param durations: List of Tuples from ... to...
    :param time_aggregation:
    :return:
    """

    query = PQL()
    for d in durations:
        if d == (None, None):
            continue
        elif d[0] is None:
            q = (
                'SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) <= "
                + str(d[1])
                + ") THEN 1 ELSE 0 END)"
            )
        elif d[1] is None:
            q = (
                'SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) >= "
                + str(d[0])
                + ") THEN 1 ELSE 0 END)"
            )
        else:
            q = (
                'SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) >= "
                + str(d[0])
                + ') AND (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) <= "
                + str(d[1])
                + ") THEN 1 ELSE 0 END)"
            )
        query.add(PQLColumn(q, str(d)))
    df_durations = dm_info.dm.get_data_frame(query)
    return df_durations


def get_quantiles_tracetime_pql(dm_info, quantiles, time_aggregation="DAYS"):
    q_quantiles = []
    for quantile in quantiles:
        q = (
            "QUANTILE(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO ALL_OCCURRENCE['Process End'], "
            'REMAP_TIMESTAMPS("'
            + dm_info.activity_table_name
            + '"."'
            + dm_info.eventtime_col
            + '", '
            + time_aggregation
            + ")), "
            + str(quantile)
            + ")"
        )
        q_quantiles.append((q, quantile))

    query = PQL()
    for q in q_quantiles:
        query.add(PQLColumn(q[0], q[1]))
    df_quantiles = dm_info.dm.get_data_frame(query)
    return df_quantiles


def get_num_cases_with_durations(dm_info, durations, time_aggregation="DAYS"):
    """Get the number of cases with the durations.

    :param dm_info:
    :param durations: List of Tuples from ... to...
    :param time_aggregation:
    :return:
    """

    query = PQL()
    for d in durations:
        if d == (None, None):
            continue
        elif d[0] is None:
            q = (
                'SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) <= "
                + str(d[1])
                + ") THEN 1 ELSE 0 END)"
            )
        elif d[1] is None:
            q = (
                'SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) >= "
                + str(d[0])
                + ") THEN 1 ELSE 0 END)"
            )
        else:
            q = (
                'SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) >= "
                + str(d[0])
                + ') AND (CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("'
                + dm_info.activity_table_name
                + '"."'
                + dm_info.eventtime_col
                + '", '
                + time_aggregation
                + ")) <= "
                + str(d[1])
                + ") THEN 1 ELSE 0 END)"
            )
        query.add(PQLColumn(q, str(d)))
    df_durations = dm_info.dm.get_data_frame(query)
    return df_durations


def get_potential_extra_bins(
    lower_end, upper_end, bin_width, num_bins, min_val, max_val
):
    """Get bins beyond the borders.

    :param lower_end:
    :param upper_end:
    :param bin_width:
    :param num_bins:
    :return:
    """
    potential_lowers = []
    potential_uppers = []
    for i in range(1, num_bins + 1):
        if lower_end - i * bin_width > min_val:
            potential_lowers.append(
                (lower_end - i * bin_width, lower_end - (i - 1) * bin_width - 1)
            )
        if upper_end + i * bin_width < max_val:
            potential_uppers.append(
                (upper_end + 1 + (i - 1) * bin_width, upper_end + i * bin_width)
            )

    return potential_lowers, potential_uppers


def choose_extra_bins(potential_lowers, potential_uppers, num_bins):
    potential_all = potential_lowers + potential_uppers
    print(potential_lowers)
    print(potential_uppers)
    if len(potential_all) == 0:
        return [], []
    extra_bins_lower = []
    extra_bins_upper = []
    take_from_upper = True
    print(num_bins)
    for i in range(num_bins):
        if (len(potential_lowers) == 0) and (len(potential_uppers) == 0):
            break
        if (
            (len(potential_lowers) > 0 and len(potential_uppers) == 0)
            or (len(potential_lowers) > 0)
            and (not take_from_upper)
        ):
            extra_bins_lower = [potential_lowers[-1]] + extra_bins_lower
            potential_lowers = potential_lowers[:-1]
            take_from_upper = True
        else:
            extra_bins_upper.append(potential_uppers[0])
            potential_uppers = potential_uppers[1:]
            take_from_upper = False
    print(extra_bins_upper)
    return extra_bins_lower, extra_bins_upper


def get_bins_trace_times(dm_info, num_bins, time_aggregation="DAYS"):
    min_percentile = 0.0
    max_percentile = 1.0
    lower_percentile = 1 / (2 * num_bins)
    upper_percentile = 1 - lower_percentile
    df_qs = get_quantiles_tracetime_pql(
        dm_info,
        [lower_percentile, upper_percentile, min_percentile, max_percentile],
        time_aggregation,
    )
    min_val = df_qs[str(min_percentile)].values[0]
    max_val = df_qs[str(max_percentile)].values[0]

    lower_end = df_qs[str(lower_percentile)].values[0]
    upper_end = df_qs[str(upper_percentile)].values[0]
    print(f"lower_end: {lower_end}, upper_end:{upper_end}")

    bin_width = int(np.ceil((upper_end - lower_end) / (num_bins - 2)))
    if (max_val - min_val + 1) / bin_width < num_bins and bin_width > 1:
        bin_width -= 1
    bins_within = (upper_end - lower_end + 1) // bin_width
    bins = [
        (lower_end + i * bin_width, lower_end + (i + 1) * bin_width - 1)
        for i in range(bins_within)
    ]
    diff_bins = num_bins - 2 - bins_within
    upper_end_within = lower_end + bin_width * bins_within - 1
    potential_lowers, potential_uppers = get_potential_extra_bins(
        lower_end, upper_end_within, bin_width, diff_bins, min_val, max_val
    )

    extra_bins_lower, extra_bins_upper = choose_extra_bins(
        potential_lowers, potential_uppers, diff_bins
    )
    if len(extra_bins_lower) > 0:
        min_inner_bin = extra_bins_lower[0][0]
    else:
        min_inner_bin = lower_end
    if len(extra_bins_upper) > 0:
        max_inner_bin = extra_bins_upper[-1][1]
    else:
        max_inner_bin = upper_end_within

    min_bin = (min_val, min_inner_bin - 1)
    max_bin = (max_inner_bin + 1, max_val)
    bins = [min_bin] + extra_bins_lower + bins + extra_bins_upper + [max_bin]
    df_histogram = get_num_cases_with_durations(
        dm_info, bins, time_aggregation=time_aggregation
    )
    df_histogram = df_histogram.transpose().reset_index()
    df_histogram.rename(
        columns={df_histogram.columns[0]: "range", df_histogram.columns[1]: "cases"},
        inplace=True,
    )
    return df_histogram
