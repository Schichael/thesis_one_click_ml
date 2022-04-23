import abc
from typing import List

import numpy as np
import pandas as pd
from ipywidgets import HBox
from ipywidgets import Layout
from ipywidgets import VBox
from ipywidgets import Widget
from plotly.graph_objs import FigureWidget

from one_click_analysis.feature_processing.attributes_new.feature import Feature
from one_click_analysis.gui.figures import AttributeDevelopmentFigure
from one_click_analysis.gui.figures import BarWithLines
from one_click_analysis.gui.figures import DistributionFigure
from one_click_analysis.gui.figures import SingleValueBox


class OverviewScreen:
    def create_box(self, traits: List[List[Widget]]):
        """
        :param traits: List of lists with the traits to display. The traits in an
        inner list are put into an HBox. The resulting HBoxes will be put into a VBox.
        :return:
        """
        vBox_overview_layout = Layout(border="2px solid gray", grid_gap="30px")
        vbox_overview = VBox(layout=vBox_overview_layout)
        boxes_traits = []
        for trait in traits:
            if len(trait) == 1 and isinstance(trait[0], FigureWidget):
                child = trait[0]
            else:
                child = HBox(children=trait)
            boxes_traits.append(child)
        vbox_overview.children = boxes_traits
        return vbox_overview

    @property
    @abc.abstractmethod
    def overview_box(self):
        pass


class OverviewScreenCaseDuration(OverviewScreen):
    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        features: List[Feature],
        target_features: List[Feature],
        timestamp_column: str,
    ):
        self.df_x = df_x
        self.df_target = df_target
        self.features = features
        self.target_features = target_features
        self.timestamp_column = timestamp_column
        self._overview_box = self._create_overview_screen()

    @property
    def overview_box(self):
        return self._overview_box

    def _create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """
        target_column_name = self.target_features[0].df_column_name
        avg_case_duration = round(self.df_target[target_column_name].mean(), 2)
        unit = self.target_features[0].unit
        # Case duration
        title = "Average case duration"
        avg_case_duration_box = SingleValueBox(
            title=title,
            val=avg_case_duration,
            unit=unit,
            title_color="Black",
            val_color="Blue",
        )
        metrics_box = HBox(
            children=[avg_case_duration_box.box],
            layout=Layout(margin="0px 30px 0px 0px"),
        )

        # development of case duration
        df_target_with_case_time = self.df_target
        df_target_with_case_time[self.timestamp_column] = self.df_x[
            self.timestamp_column
        ]

        fig_case_duration_development = AttributeDevelopmentFigure(
            df=df_target_with_case_time,
            time_col=self.timestamp_column,
            attribute_cols=self.target_features[0].df_column_name,
            fill=True,
            title="Case duration development",
        )

        # case duration distribution
        fig_distribution = DistributionFigure(
            df=self.df_target,
            attribute_col=self.target_features[0].df_column_name,
            attribute_name="Case duration",
            num_bins=10,
        )

        return self.create_box(
            [
                [metrics_box],
                [fig_case_duration_development.figure],
                [fig_distribution.figure],
            ]
        )


class OverviewScreenDecisionRules(OverviewScreen):
    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        features: List[Feature],
        target_features: List[Feature],
        timestamp_column: str,
        source_activity: str,
        case_duration_col_name: str,
    ):
        self.df_x = df_x
        self.df_target = df_target
        self.features = features
        self.target_features = target_features
        self.timestamp_column = timestamp_column
        self.source_activity = source_activity
        self.case_duration_col_name = case_duration_col_name
        self.target_activities = self._get_target_activities()
        self._overview_box = self._create_overview_screen()

    @property
    def overview_box(self):
        return self._overview_box

    def _get_target_activities(self):
        target_activities = []
        for tf in self.target_features:
            target_activities.append(tf.attribute_value)
        return target_activities

    def _create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """

        transitions_with_source_activity = len(self.df_target.index)
        title = f"Transitions from activity '{self.source_activity}'"
        cases_with_activity_box = SingleValueBox(
            title=title,
            val=transitions_with_source_activity,
            title_color="Black",
            val_color="Blue",
        )

        target_column_names = [x.df_column_name for x in self.target_features]
        # Get average case durations
        avg_case_durations = []
        for col_name in target_column_names:
            if len(self.df_target[self.df_target[col_name] == 1].index) == 0:
                avg_case_durations.append(0)
            else:
                # self.activity_case_key
                df_with_target = self.df_x[self.df_target[col_name] == 1]
                df_grouped = df_with_target.groupby(level=0).first()
                avg_case_durations.append(
                    round(
                        df_grouped[self.case_duration_col_name].mean(),
                        2,
                    )
                )

        num_datapoints_with_target = []
        for tf in self.target_features:

            num_datapoints_with_target.append(tf.metrics["case_count"])

        # barplot with cases with target activities and metric line plot

        barplot_args = {
            "x": self.target_activities,
            "y": num_datapoints_with_target,
            "name": "Cases with transition",
        }
        line_plot_args = {
            "x": self.target_activities,
            "y": avg_case_durations,
            "name": "Average case duration",
        }

        layout_args = {
            "xaxis_title": "Transitions to",
            "yaxis_title": "Cases with transition",
            "yaxis2_title": "Average case duration [Days]",
            "title": "Cases with transitions and average case duration",
        }

        barplot = BarWithLines(barplot_args, line_plot_args, **layout_args)

        # development of case duration
        title_transition_development = (
            "Cases with transitions from "
            + self.source_activity
            + " to selected target activities"
        )
        df_target_with_case_time = self.df_target
        df_target_with_case_time[self.timestamp_column] = self.df_x[
            self.timestamp_column
        ]
        fig_transition_development = AttributeDevelopmentFigure(
            df=df_target_with_case_time,
            time_col=self.timestamp_column,
            attribute_cols=target_column_names,
            attribute_names=self.target_activities,
            fill=False,
            title=title_transition_development,
            data_aggregation=np.sum,
        )
        return self.create_box(
            [
                [cases_with_activity_box.box],
                [barplot.figure],
                [fig_transition_development.figure],
            ]
        )
