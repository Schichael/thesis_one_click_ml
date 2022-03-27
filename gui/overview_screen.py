from typing import List
from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Box
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox
from pycelonis.celonis_api.pql.pql import PQL
from pycelonis.celonis_api.pql.pql import PQLColumn

from feature_processing.feature_processor import FeatureProcessor

# TODO: Make the OverviewBox more general or add an abstract overview class from
# which the overview screens for the different analyses can inherent common
# methods.


class OverviewScreen:
    def __init__(self, fp: FeatureProcessor):
        self.fp = fp

    def get_overview_box(self):
        vBox_overview_layout = Layout(border="2px solid gray", grid_gap="30px")
        vBox_overview = VBox(layout=vBox_overview_layout)
        avg_case_duration = get_case_duration_pql(self.fp)
        # Case duration

        avg_case_duration_box = HBox(
            [
                Box(
                    [
                        HTML(
                            '<center><span style="font-weight:bold"> Average Case '
                            "Duration</span><br><span "
                            'style="color: Red; font-size:16px">'
                            + str(round(avg_case_duration))
                            + "\xa0"
                            + self.fp.label.unit
                            + "</span></center>"
                        )
                    ],
                    layout=Layout(
                        border="3px double CornflowerBlue", margin="20px 50px 0px 10px"
                    ),
                )
            ],
            layout=Layout(margin="0px 30px 0px 0px"),
        )

        # development of case duration
        df_case_duration_dev = get_case_duration_development_pql(self.fp)
        fig_case_duration_development = px.area(
            df_case_duration_dev,
            x="datetime",
            y="case duration",
            title="Case duration development",
            height=250,
        )
        fig_case_duration_development.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        f_widget_case_duration_dev = go.FigureWidget(fig_case_duration_development)
        # case duration distribution
        df_distribution = get_bins_trace_times(self.fp, 10, time_aggregation="DAYS")
        fig_distribution = px.bar(
            df_distribution,
            x="range",
            y="cases",
            title="Case duration distribution",
            height=300,
        )
        fig_distribution.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        f_widget_distribution = go.FigureWidget(fig_distribution)
        vBox_overview.children = [
            avg_case_duration_box,
            f_widget_case_duration_dev,
            f_widget_distribution,
        ]
        return vBox_overview


# TODO:Some of the following PQL queries could also be moved to the FeatureProcessor.


def get_case_duration_pql(
    fp: FeatureProcessor, aggregation: str = "AVG", time_aggregation: str = "DAYS"
):
    q = (
        aggregation + "(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO "
        "ALL_OCCURRENCE["
        "'Process End'], REMAP_TIMESTAMPS(\""
        + fp.activity_table_name
        + '"."'
        + fp.eventtime_col
        + '", '
        "" + time_aggregation + ")))"
    )
    query = PQL()
    query.add(PQLColumn(q, "average case duration"))
    df_avg_case_duration = fp.dm.get_data_frame(query)
    return df_avg_case_duration["average case duration"].values[0]


def get_case_duration_development_pql(
    fp: FeatureProcessor,
    duration_aggregation: str = "AVG",
    date_aggregation: str = "ROUND_MONTH",
    time_aggregation: str = "DAYS",
):
    q_date = (
        date_aggregation
        + '("'
        + fp.activity_table_name
        + '"."'
        + fp.eventtime_col
        + '")'
    )
    q_duration = (
        duration_aggregation + "(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] "
        "TO ALL_OCCURRENCE["
        "'Process End'], REMAP_TIMESTAMPS(\""
        + fp.activity_table_name
        + '"."'
        + fp.eventtime_col
        + '", '
        "" + time_aggregation + ")))"
    )
    query = PQL()
    query.add(PQLColumn(q_date, "datetime"))
    query.add(PQLColumn(q_duration, "case duration"))
    df_avg_case_duration = fp.dm.get_data_frame(query)
    return df_avg_case_duration


def get_quantiles_tracetime_pql(fp, quantiles, time_aggregation="DAYS"):
    q_quantiles = []
    for quantile in quantiles:
        q = (
            "QUANTILE(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO "
            "ALL_OCCURRENCE['Process End'], "
            'REMAP_TIMESTAMPS("'
            + fp.activity_table_name
            + '"."'
            + fp.eventtime_col
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
    df_quantiles = fp.dm.get_data_frame(query)
    return df_quantiles


def get_num_cases_with_durations(
    fp: FeatureProcessor,
    durations: List[Tuple[int, int]],
    time_aggregation: str = "DAYS",
):
    """Get the number of cases with the durations.

    :param fp:
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
                "SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, "
                'REMAP_TIMESTAMPS("'
                + fp.activity_table_name
                + '"."'
                + fp.eventtime_col
                + '", '
                + time_aggregation
                + ")) <= "
                + str(d[1])
                + ") THEN 1 ELSE 0 END)"
            )
        elif d[1] is None:
            q = (
                "SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, "
                'REMAP_TIMESTAMPS("'
                + fp.activity_table_name
                + '"."'
                + fp.eventtime_col
                + '", '
                + time_aggregation
                + ")) >= "
                + str(d[0])
                + ") THEN 1 ELSE 0 END)"
            )
        else:
            q = (
                "SUM(CASE WHEN (CALC_THROUGHPUT(CASE_START TO CASE_END, "
                'REMAP_TIMESTAMPS("'
                + fp.activity_table_name
                + '"."'
                + fp.eventtime_col
                + '", '
                + time_aggregation
                + ")) >= "
                + str(d[0])
                + ") AND (CALC_THROUGHPUT(CASE_START TO CASE_END, "
                'REMAP_TIMESTAMPS("'
                + fp.activity_table_name
                + '"."'
                + fp.eventtime_col
                + '", '
                + time_aggregation
                + ")) <= "
                + str(d[1])
                + ") THEN 1 ELSE 0 END)"
            )
        query.add(PQLColumn(q, str(d)))
    df_durations = fp.dm.get_data_frame(query)
    return df_durations


def get_potential_extra_bins(
    lower_end: int,
    upper_end: int,
    bin_width: int,
    num_bins: int,
    min_val: int,
    max_val: int,
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


def choose_extra_bins(
    potential_lowers: List[Tuple[int, int]],
    potential_uppers: List[Tuple[int, int]],
    num_bins: int,
):
    potential_all = potential_lowers + potential_uppers

    if len(potential_all) == 0:
        return [], []
    extra_bins_lower = []
    extra_bins_upper = []
    take_from_upper = True
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
    return extra_bins_lower, extra_bins_upper


def get_bins_trace_times(
    fp: FeatureProcessor, num_bins: int, time_aggregation: str = "DAYS"
):
    min_percentile = 0.0
    max_percentile = 1.0
    lower_percentile = 1 / (2 * num_bins)
    upper_percentile = 1 - lower_percentile
    df_qs = get_quantiles_tracetime_pql(
        fp,
        [lower_percentile, upper_percentile, min_percentile, max_percentile],
        time_aggregation,
    )
    min_val = df_qs[str(min_percentile)].values[0]
    max_val = df_qs[str(max_percentile)].values[0]

    lower_end = df_qs[str(lower_percentile)].values[0]
    upper_end = df_qs[str(upper_percentile)].values[0]

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
        fp, bins, time_aggregation=time_aggregation
    )
    df_histogram = df_histogram.transpose().reset_index()
    df_histogram.rename(
        columns={df_histogram.columns[0]: "range", df_histogram.columns[1]: "cases"},
        inplace=True,
    )
    return df_histogram
