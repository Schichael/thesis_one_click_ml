from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Box
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

from one_click_analysis.feature_processing.feature_processor import FeatureProcessor

# TODO: Make the OverviewBox more general or add an abstract overview class from
# which the overview screens for the different analyses can inherent common
# methods.


class OverviewScreen:
    def __init__(self, fp: FeatureProcessor):
        """

        :param fp: FeatureProcessor with processed features
        """
        self.fp = fp
        self.overview_box = None

    def create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """
        vBox_overview_layout = Layout(border="2px solid gray", grid_gap="30px")
        vBox_overview = VBox(layout=vBox_overview_layout)
        avg_case_duration = get_avg_case_duration(self.fp)
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
                            + self.fp.labels[0].unit
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
        df_case_duration_dev = get_case_duration_development(self.fp)
        fig_case_duration_development = px.area(
            df_case_duration_dev,
            x="Case start time",
            y=self.fp.labels[0].df_attribute_name,
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
        df_distribution = compute_binned_distribution_case_durations(self.fp, 10)
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
        self.overview_box = vBox_overview


# TODO:Some of the following PQL queries could also be moved to the FeatureProcessor
#  or obtained from the df in the FeatureProcessor object.
#  They should also be made more generic.


def get_avg_case_duration(fp: FeatureProcessor) -> float:
    """Get the aggregates case duration of the cases

    :param fp: FeatureProcessor with the processed features
    :param aggregation: aggregation for the case duration, e.g. 'AVG' to get the
    average case duration
    :param time_aggregation: aggregation for the time, e.g. 'DAYS' or 'MONTHS'
    :return: aggregated case duration
    """

    label_column_name = fp.labels[0].df_attribute_name
    avg_case_duration = fp.df[label_column_name].mean()
    return avg_case_duration


def get_case_duration_development(
    fp: FeatureProcessor,
) -> pd.DataFrame:
    """Get aggregated case duration aggregated over a time period

    :param fp: FeatureProcessor with the processed features
    :return: DataFrame with the aggregated case duration aggregated over time
    """

    df = fp.df[["caseid", "Case start time", fp.labels[0].df_attribute_name]].copy()
    df["Case start time"] = df["Case start time"].dt.to_period("M").astype(str)
    num_cases_all_df = df.groupby("Case start time", as_index=False)[
        fp.labels[0].df_attribute_name
    ].mean()

    return num_cases_all_df


def get_quantiles_case_duration_pql(
    fp: FeatureProcessor, quantiles: List[float]
) -> dict:
    """

    :param fp: FeatureProcessor with the processed features
    :param quantiles: list of quantiles for which to get the values
    :return: DataFrame with the case durations of the quantile values
    """

    quantile_dict = {}
    for q in quantiles:
        quantile_dict[q] = fp.df[fp.labels[0].df_attribute_name].quantile(q)

    return quantile_dict


def get_num_cases_with_durations(
    fp: FeatureProcessor,
    duration_intervals: List[Tuple[int, int]],
) -> pd.DataFrame:
    """Get the number of cases with the specified duration intervals.

    :param fp: FeatureProcessor with the processed features
    :param duration_intervals: List of duration intervals from <first value> to <second
    value>
    :return: DataFrame with number of cases per duration interval
    """

    num_cases_dict = {}
    for d in duration_intervals:
        if d == (None, None):
            continue
        elif d[0] is None:
            num_cases_dict[str(d)] = [
                len(fp.df[fp.df[fp.labels[0].df_attribute_name] <= d[1]].index)
            ]

        elif d[1] is None:
            num_cases_dict[str(d)] = [
                len(fp.df[fp.df[fp.labels[0].df_attribute_name] >= d[0]].index)
            ]
        else:
            num_cases_dict[str(d)] = [
                len(
                    fp.df[
                        (fp.df[fp.labels[0].df_attribute_name] >= d[0])
                        & (fp.df[fp.labels[0].df_attribute_name] <= d[1])
                    ].index
                )
            ]
    df = pd.DataFrame(data=num_cases_dict)
    return df


def get_potential_extra_bins(
    lower_end: int,
    upper_end: int,
    bin_width: int,
    num_bins: int,
    min_val: int,
    max_val: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Get potential bins beyond the bins that are already defined.

    :param lower_end: lower value of the bins that are already defined
    :param upper_end: upper value of the bins that are already defined
    :param bin_width: width of a bin
    :param num_bins: desired number of bins to add
    :param min_val: minimum value of the considered data
    :param max_val: maximum value of the considered data
    :return: lists with lower and upper values of potential extra bins
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
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Choose extra bins beyond the original bins from the potential bins defined in
    potential_lowers and potential_uppers.

    :param potential_lowers: potential lower values for extra bins
    :param potential_uppers: potential upper values for extra bins
    :param num_bins: desired number of extra bins
    :return: lists with lower and upper values of chosen extra bins
    """
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


def compute_binned_distribution_case_durations(fp: FeatureProcessor, num_bins: int):
    """Compute a binned distribution for case durations. The binning is done using
    equal width binning for all inner bins. The outer bins at the beginning and the
    end contain outliers.

    :param fp: FeatureProcessor with the processed features
    :param num_bins: number of desired bins
    :return: DataFrame with the case duration distribution
    """
    min_percentile = 0.0
    max_percentile = 1.0
    lower_percentile = 1 / (2 * num_bins)
    upper_percentile = 1 - lower_percentile

    qs_label_val = get_quantiles_case_duration_pql(
        fp, [lower_percentile, upper_percentile, min_percentile, max_percentile]
    )
    min_val = qs_label_val[min_percentile]
    max_val = qs_label_val[max_percentile]

    lower_end = qs_label_val[lower_percentile]
    upper_end = qs_label_val[upper_percentile]

    bin_width = int(np.ceil((upper_end - lower_end) / (num_bins - 2)))
    if (max_val - min_val + 1) / bin_width < num_bins and bin_width > 1:
        bin_width -= 1
    bins_within = int((upper_end - lower_end + 1) // bin_width)
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

    if min_inner_bin == min_val:
        min_bin = []
    else:
        min_bin = [(min_val, min_inner_bin - 1)]

    if max_inner_bin == max_val:
        max_bin = []
    else:
        max_bin = [(max_inner_bin + 1, max_val)]

    bins = min_bin + extra_bins_lower + bins + extra_bins_upper + max_bin
    df_histogram = get_num_cases_with_durations(fp, bins)
    df_histogram = df_histogram.transpose().reset_index()
    df_histogram.rename(
        columns={df_histogram.columns[0]: "range", df_histogram.columns[1]: "cases"},
        inplace=True,
    )
    return df_histogram
