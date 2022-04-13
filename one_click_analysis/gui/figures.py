import abc
from typing import List
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout

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
        self.title_colot = title_color
        self.val_color = val_color
        self.box = self._crate_box()

    def _crate_box(self):
        unit = "" if self.unit is None else self.unit
        html = HTML(
            '<center><span style="font-weight:bold"> Average Case '
            "Duration</span><br><span "
            'style="color: Red; font-size:16px">'
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
        attribute_names: Optional[List[str]] = None,
        time_aggregation: Optional[str] = "M",
        fill: bool = False,
        **kwargs
    ):
        """

        :param df: DataFrame
        :param time_col: name of the time column
        :param attribute_cols: names of the attribute columns
        :param attribute_names: the names of the attributes for the legend
        :param time_aggregation: how to aggregate the time. One of pandas’ offset
        strings or an Offset object. E.g. 'Y' for yearly, 'M' for monthly, 'D' for
        daily
        :param fill: whether to fille the plot to become an area chart. If True,
        the plot for the last attribute in attribute_cols will be filled to zero_y.
        :param kwargs: arguments to use for the figure layout
        """
        # default layout vals for this figure
        layout_vals_this_fig = {"title": "Attribute development"}
        layout_vals_this_fig.update(kwargs)
        super().__init__(**layout_vals_this_fig)
        self.df = df
        self.time_col = time_col
        self.attribute_cols = utils.make_list(attribute_cols)
        self.attribute_names = attribute_names
        self.time_aggregation = time_aggregation
        self.fill = fill
        self.figure = self._create_figure()

    def _create_figure(self):
        df = self.df[[self.time_col] + self.attribute_cols].copy()
        df["time_agg"] = (
            df[self.time_col].dt.to_period(self.time_aggregation).astype(str)
        )
        df = df.groupby("time_agg", as_index=False)[self.attribute_cols].mean()

        fig = go.Figure(layout_title_text=self.layout_vals["title"])

        for i, attribute in enumerate(self.attribute_cols):
            if self.attribute_names is not None:
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
