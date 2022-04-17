import abc
from typing import List

import numpy as np
from ipywidgets import HBox
from ipywidgets import Layout
from ipywidgets import VBox
from ipywidgets import Widget
from plotly.graph_objs import FigureWidget

from one_click_analysis.feature_processing.feature_processor import FeatureProcessor
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
    def __init__(self, fp: FeatureProcessor):
        self.fp = fp

        self._overview_box = self._create_overview_screen()

    @property
    def overview_box(self):
        return self._overview_box

    def _create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """
        label_column_name = self.fp.labels[0].df_attribute_name
        avg_case_duration = round(self.fp.df[label_column_name].mean(), 2)
        unit = self.fp.labels[0].unit
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
        fig_case_duration_development = AttributeDevelopmentFigure(
            df=self.fp.df,
            time_col="Case start time",
            attribute_cols=self.fp.labels[0].df_attribute_name,
            fill=True,
            title="Case duration development",
        )

        # case duration distribution
        fig_distribution = DistributionFigure(
            df=self.fp.df,
            attribute_col=self.fp.labels[0].df_attribute_name,
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
    def __init__(self, fp, source_activity: str, target_activities: List[str]):
        self.fp = fp
        self.source_activity = source_activity
        self.target_activities = target_activities
        self._overview_box = self._create_overview_screen()

    @property
    def overview_box(self):
        return self._overview_box

    def _create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """

        cases_with_source_activity = len(self.fp.df.index)
        title = "Cases with activity " + self.source_activity
        cases_with_activity_box = SingleValueBox(
            title=title,
            val=cases_with_source_activity,
            title_color="Black",
            val_color="Blue",
        )

        label_column_names = [x.df_attribute_name for x in self.fp.labels]
        # Get average case durations
        avg_case_durations = []
        for col_name in label_column_names:
            if len(self.fp.df[self.fp.df[col_name] == 1].index) == 0:
                avg_case_durations.append(0)
            else:
                avg_case_durations.append(
                    round(
                        self.fp.df[self.fp.df[col_name] == 1][
                            "case " "duration"
                        ].mean(),
                        2,
                    )
                )

        num_cases_with_label = []
        for col_name in label_column_names:

            num_cases_with_label.append(
                len(self.fp.df[self.fp.df[col_name] == 1].index)
            )

        # barplot with cases with target activities and metric line plot

        barplot_args = {
            "x": self.target_activities,
            "y": num_cases_with_label,
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
            + " to selected target "
            "activities"
        )
        fig_transition_development = AttributeDevelopmentFigure(
            df=self.fp.df,
            time_col="Case start time",
            attribute_cols=label_column_names,
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
