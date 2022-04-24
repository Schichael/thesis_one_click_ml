import abc
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from plotly.subplots import make_subplots

from one_click_analysis import utils


class SingleValueBox:
    def __init__(
        self,
        title: str,
        val: float,
        unit: Optional[str] = None,
        title_color: Optional[str] = None,
        val_color: Optional[str] = None,
    ):
        self.title = title
        self.val = val
        self.unit = unit
        self.title_color = title_color
        self.val_color = val_color
        self.box = self._crate_box()

    def _crate_box(self):
        unit = "" if self.unit is None else self.unit
        html = HTML(
            '<center><span style="font-weight:bold; color: '
            + self.title_color
            + '"> '
            + self.title
            + '</span><br><span style="color: '
            + self.val_color
            + '; font-size:16px">'
            + str(self.val)
            + "\xa0"
            + unit
            + "</span></center>"
        )
        box = HBox(
            children=[html],
            layout=Layout(
                border="3px double CornflowerBlue", margin="20px 50px 0px 10px"
            ),
        )
        return box


class Figure(abc.ABC):
    def __init__(self, **kwargs):
        """

        :param kwargs: arguments to use for the figure layout
        """
        self.layout_vals = {
            "title": "",
            "height": 250,
            "xaxis_title": None,
            "yaxis_title": None,
            "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        }
        # Update layout vals with kwargs
        self.layout_vals.update(kwargs)


class AttributeDevelopmentFigure(Figure):
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        attribute_cols: List[str] or str,
        attribute_names: Optional[List[str] or str] = None,
        time_aggregation: Optional[str] = "M",
        data_aggregation: Union[str, Callable] = "mean",
        fill: bool = False,
        **kwargs,
    ):
        """

        :param df: DataFrame
        :param time_col: name of the time column
        :param attribute_cols: names of the attribute columns
        :param attribute_names: the names of the attributes for the legend
        :param time_aggregation: how to aggregate the time. One of pandas’ offset
        strings or an Offset object. E.g. 'Y' for yearly, 'M' for monthly, 'D' for
        daily
        :param data_Aggregation: the way to aggregate the data
        :param fill: whether to fille the plot to become an area chart. If True,
        the plot for the last attribute in attribute_cols will be filled to zero_y.
        :param kwargs: arguments to use for the figure layout
        """
        # default layout vals for this figure

        self.df = df
        self.time_col = time_col
        self.attribute_cols = utils.make_list(attribute_cols)
        self.attribute_names = utils.make_list(attribute_names)
        self.time_aggregation = time_aggregation
        self.data_aggregation = data_aggregation
        self.fill = fill
        layout_vals_this_fig = self._get_layout_args(**kwargs)
        super().__init__(**layout_vals_this_fig)
        self.figure = self._create_figure()

    def _get_layout_args(self, **kwargs):
        # Default vals
        layout_args = {"title": "Attribute development"}
        layout_args.update(kwargs)
        return layout_args

    def _create_figure(self):
        df = self.df[[self.time_col] + self.attribute_cols].copy()
        df["time_agg"] = (
            df[self.time_col].dt.to_period(self.time_aggregation).astype(str)
        )
        df = df.groupby("time_agg", as_index=False)[self.attribute_cols].aggregate(
            self.data_aggregation
        )

        fig = go.Figure(layout_title_text=self.layout_vals["title"])

        for i, attribute in enumerate(self.attribute_cols):
            if self.attribute_names:
                name = self.attribute_names[i]
            else:
                name = attribute

            if self.fill:
                if i < len(self.attribute_cols) - 1:
                    fill = "tonexty"
                else:
                    fill = "tozeroy"
            else:
                fill = None

            fig.add_trace(
                go.Scatter(
                    x=df["time_agg"],
                    y=df[attribute],
                    fill=fill,
                    name=name,
                )
            )

        fig.update_layout(**self.layout_vals)
        fig_widget = go.FigureWidget(fig)
        return fig_widget


class BarWithLines(Figure):
    def __init__(self, barplot_args: dict, line_plot_args: dict, **kwargs):
        self.barplot_args = barplot_args
        self.line_plot_args = line_plot_args
        layout_args = self._get_layout_args(**kwargs)
        super().__init__(**layout_args)
        self.figure = self._create_figure()

    def _create_figure(self):
        trace_bar = go.Bar(self.barplot_args, marker=dict(color="rgb(34,163,192)"))
        trace_metric = go.Scatter(
            self.line_plot_args, mode="lines", marker={"color": "red"}
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(trace_bar, secondary_y=False)
        fig.add_trace(trace_metric, secondary_y=True)

        fig.update_layout(self.layout_vals)
        fig_widget = go.FigureWidget(fig)
        return fig_widget

    def _get_layout_args(self, **kwargs):
        layout_args = kwargs
        return layout_args


class NumericalAttributeEffectOnLabel(Figure):
    def __init__(
        self,
        df: pd.DataFrame,
        attribute_col: str,
        attribute_name: str,
        attribute_aggr,
    ):
        pass


class DistributionFigure(Figure):
    def __init__(
        self,
        df: pd.DataFrame,
        attribute_col: str,
        attribute_name: Optional[str],
        num_bins: int,
        **kwargs,
    ):
        self.df = df
        self.attribute_col = attribute_col
        self.attribute_name = attribute_name
        self.num_bins = num_bins
        layout_args = self._get_layout_args(**kwargs)
        super().__init__(**layout_args)
        self.figure = self._create_figure()

    def _get_layout_args(self, **kwargs):
        # Default args
        attr_name = (
            self.attribute_name
            if self.attribute_name is not None
            else self.attribute_col
        )
        layout_args = {"title": f"{attr_name} distribution"}
        layout_args.update(kwargs)
        return layout_args

    def _create_figure(self):
        df_distribution = self.compute_binned_distribution_case_durations(
            df=self.df, attribute_col=self.attribute_col, num_bins=self.num_bins
        )
        fig_distribution = px.bar(df_distribution, x="range", y="cases")
        fig_distribution.update_layout(**self.layout_vals)
        fig_widget = go.FigureWidget(fig_distribution)
        return fig_widget

    def compute_binned_distribution_case_durations(
        self, df: pd.DataFrame, attribute_col: str, num_bins: int
    ):
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

        qs_label_val = self.get_quantiles_case_duration_pql(
            df,
            attribute_col,
            [lower_percentile, upper_percentile, min_percentile, max_percentile],
        )
        min_val = qs_label_val[min_percentile]
        max_val = qs_label_val[max_percentile]

        lower_end = round(qs_label_val[lower_percentile])
        upper_end = round(qs_label_val[upper_percentile])

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
        potential_lowers, potential_uppers = self.get_potential_extra_bins(
            lower_end, upper_end_within, bin_width, diff_bins, min_val, max_val
        )

        extra_bins_lower, extra_bins_upper = self.choose_extra_bins(
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
        df_histogram = self.get_num_cases_with_durations(df, attribute_col, bins)
        df_histogram = df_histogram.transpose().reset_index()
        df_histogram.rename(
            columns={
                df_histogram.columns[0]: "range",
                df_histogram.columns[1]: "cases",
            },
            inplace=True,
        )
        return df_histogram

    def choose_extra_bins(
        self,
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

    def get_potential_extra_bins(
        self,
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

    def get_num_cases_with_durations(
        self,
        df: pd.DataFrame,
        attribute_col: str,
        duration_intervals: List[Tuple[int, int]],
    ) -> pd.DataFrame:
        """Get the number of cases with the specified duration intervals.

        :param df:
        :param attribute_col:
        :param duration_intervals: List of duration intervals from <first value> to
        <second
        value>
        :return: DataFrame with number of cases per duration interval
        """

        num_cases_dict = {}
        for d in duration_intervals:
            if d == (None, None):
                continue
            elif d[0] is None:
                num_cases_dict[str(d)] = [len(df[df[attribute_col] <= d[1]].index)]

            elif d[1] is None:
                num_cases_dict[str(d)] = [len(df[df[attribute_col] >= d[0]].index)]
            else:
                num_cases_dict[str(d)] = [
                    len(
                        df[
                            (df[attribute_col] >= d[0]) & (df[attribute_col] <= d[1])
                        ].index
                    )
                ]
        df = pd.DataFrame(data=num_cases_dict)
        return df

    def get_quantiles_case_duration_pql(
        self, df: pd.DataFrame, attribute_col: str, quantiles: List[float]
    ) -> dict:
        """

        :param df:
        :param attribute_col:
        :param quantiles: list of quantiles for which to get the values
        :return: DataFrame with the case durations of the quantile values
        """

        quantile_dict = {}
        for q in quantiles:
            quantile_dict[q] = df[attribute_col].quantile(q)

        return quantile_dict
