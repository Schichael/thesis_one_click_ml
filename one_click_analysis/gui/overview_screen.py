import abc
from typing import List

import numpy as np
import pandas as pd
from ipywidgets import HBox
from ipywidgets import Layout
from ipywidgets import VBox
from ipywidgets import Widget
from plotly.graph_objs import FigureWidget

from one_click_analysis import utils
from one_click_analysis.feature_processing.attributes.feature import Feature
from one_click_analysis.gui.figures import AttributeDevelopmentFigure
from one_click_analysis.gui.figures import BarWithLines
from one_click_analysis.gui.figures import DistributionFigure
from one_click_analysis.gui.figures import GroupedBar
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
        num_cases: int,
    ):
        self.df_x = df_x
        self.df_target = df_target
        self.features = features
        self.target_features = target_features
        self.timestamp_column = timestamp_column
        self.num_cases = num_cases
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

        title = "Number of selected cases"
        num_cases_box = SingleValueBox(
            title=title,
            val=self.num_cases,
            unit=None,
            title_color="Black",
            val_color="Blue",
        )

        metrics_box = HBox(
            children=[avg_case_duration_box.box, num_cases_box.box],
            layout=Layout(margin="0px 30px 0px 0px"),
        )

        # development of case duration
        df_target_with_case_time = self.df_target
        df_target_with_case_time[self.timestamp_column] = self.df_x[
            self.timestamp_column
        ]
        fig_case_duration_development_args = {
            "xaxis_title": "Date",
            "yaxis_title": "Average case duration (Days)",
        }

        fig_case_duration_development = AttributeDevelopmentFigure(
            df=df_target_with_case_time,
            time_col=self.timestamp_column,
            attribute_cols=self.target_features[0].df_column_name,
            fill=True,
            case_level=False,
            title="Case duration development",
            **fig_case_duration_development_args,
        )

        fig_distribution_args = {
            "xaxis_title": "Case duration (Days)",
            "yaxis_title": "Number of cases",
        }
        # case duration distribution
        fig_distribution = DistributionFigure(
            df=self.df_target,
            attribute_col=self.target_features[0].df_column_name,
            attribute_name="Case duration",
            num_bins=10,
            **fig_distribution_args,
        )

        return self.create_box(
            [
                [metrics_box],
                [fig_case_duration_development.figure],
                [fig_distribution.figure],
            ]
        )


class OverviewScreenRoutingDecisions(OverviewScreen):
    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        features: List[Feature],
        target_features: List[Feature],
        timestamp_column: str,
        source_activity: str,
        case_duration_col_name: str,
        num_cases: int,
    ):
        self.df_x = df_x
        self.df_target = df_target
        self.features = features
        self.target_features = target_features
        self.timestamp_column = timestamp_column
        self.source_activity = source_activity
        self.case_duration_col_name = case_duration_col_name
        self.num_cases = num_cases
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
        # Transitions from source activity to selected target activities. It is the
        # length of the DataFrame without the rows where there is no target activity
        # (Then after the source activity none of the target activities occur.)
        # Get indices where there is a decision to one of the target activities
        indices_with_target_activities = []
        num_transitions = 0
        for target_feature in self.target_features:
            col_name = target_feature.df_column_name
            idxs_with_target = np.where(self.df_target[col_name] == 1)[0]
            num_transitions += len(idxs_with_target)
            indices_with_target_activities = (
                indices_with_target_activities + idxs_with_target.tolist()
            )

        title = (
            f"Total number of decisions from activity '{self.source_activity}' "
            f"to selected target activities"
        )
        number_transitions_box = SingleValueBox(
            title=title,
            val=num_transitions,
            title_color="Black",
            val_color="Blue",
        )

        # Total number of selected cases
        title = "Number of selected cases"
        num_cases_box = SingleValueBox(
            title=title,
            val=self.num_cases,
            unit=None,
            title_color="Black",
            val_color="Blue",
        )

        # Number of cases with at least one transition from source activity to one of
        # the target activities
        num_cases_with_transitions = len(
            set(
                self.df_target.iloc[
                    indices_with_target_activities
                ].index.get_level_values(0)
            )
        )
        title = (
            f"Total number of cases with decisions from activity "
            f"'{self.source_activity}' "
            f"to selected target activities"
        )
        num_cases_with_transition_box = SingleValueBox(
            title=title,
            val=num_cases_with_transitions,
            unit=None,
            title_color="Black",
            val_color="Blue",
        )

        metrics_box = HBox(
            children=[
                num_cases_box.box,
                num_cases_with_transition_box.box,
                number_transitions_box.box,
            ],
            layout=Layout(margin="0px 30px 0px 0px"),
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

        num_cases_with_target = []
        for tf in self.target_features:

            num_cases_with_target.append(tf.metrics["case_count"])

        # Sort for number of cases with target
        print(f"num_cases_with_target: {num_cases_with_target}")
        sorted_indices = np.argsort(np.argsort(num_cases_with_target))
        print(f"sorted indices: {sorted_indices}")
        num_cases_with_target_sorted = [None] * sorted_indices.size
        avg_case_durations_sorted = [None] * sorted_indices.size
        target_activities_sorted = [None] * sorted_indices.size

        for i, el in enumerate(sorted_indices):
            num_cases_with_target_sorted[el] = num_cases_with_target[i]
            avg_case_durations_sorted[el] = avg_case_durations[i]
            target_activities_sorted[el] = self.target_activities[i]
        num_cases_with_target_sorted.reverse()
        avg_case_durations_sorted.reverse()
        target_activities_sorted.reverse()
        # barplot with cases with target activities and metric line plot

        barplot_args = {
            "x": target_activities_sorted,
            "y": num_cases_with_target_sorted,
            "name": "Cases with transition",
        }
        line_plot_args = {
            "x": target_activities_sorted,
            "y": avg_case_durations_sorted,
            "name": "Average case duration",
        }

        layout_args = {
            "xaxis_title": "Decisions to",
            "yaxis_title": "Cases with decision",
            "yaxis2_title": "Average case duration [Days]",
            "title": "Cases containing decisions and average case duration (One case "
            "can have decisions to more than one activity)",
        }

        barplot = BarWithLines(barplot_args, line_plot_args, **layout_args)

        # development of case duration
        title_transition_development = (
            "Cases with decisions from "
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
            case_level=True,
            case_level_aggregation="max",
            title=title_transition_development,
            data_aggregation=np.sum,
        )
        return self.create_box(
            [
                [metrics_box],
                [barplot.figure],
                [fig_transition_development.figure],
            ]
        )


class OverviewScreenTransitionTime(OverviewScreen):
    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        features: List[Feature],
        target_features: List[Feature],
        timestamp_column: str,
        source_activity: str,
        target_activity: str,
        num_cases: int,
    ):
        self.df_x = df_x
        self.df_target = df_target
        self.features = features
        self.target_features = target_features
        self.timestamp_column = timestamp_column
        self.source_activity = source_activity
        self.target_activity = target_activity
        self.num_cases = num_cases
        self._overview_box = self._create_overview_screen()

    @property
    def overview_box(self):
        return self._overview_box

    def _create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """
        target_column_name = self.target_features[0].df_column_name
        avg_transition_duration = round(self.df_target[target_column_name].mean(), 2)
        unit = self.target_features[0].unit
        # Case duration
        title = "Average transition duration"
        avg_tansition_duration_box = SingleValueBox(
            title=title,
            val=avg_transition_duration,
            unit=unit,
            title_color="Black",
            val_color="Blue",
        )

        num_transitions = len(self.df_target.index)
        title = f"Transitions from '{self.source_activity}' to '{self.target_activity}'"
        num_transitions_box = SingleValueBox(
            title=title,
            val=num_transitions,
            title_color="Black",
            val_color="Blue",
        )

        title = "Number of selected cases"
        num_cases_box = SingleValueBox(
            title=title,
            val=self.num_cases,
            unit=None,
            title_color="Black",
            val_color="Blue",
        )

        metrics_box = HBox(
            children=[
                avg_tansition_duration_box.box,
                num_transitions_box.box,
                num_cases_box.box,
            ],
            layout=Layout(margin="0px 30px 0px 0px"),
        )

        # development of transition duration
        df_target_with_transition_time = self.df_target
        df_target_with_transition_time[self.timestamp_column] = self.df_x[
            self.timestamp_column
        ]

        fig_transition_duration_development = AttributeDevelopmentFigure(
            df=df_target_with_transition_time,
            time_col=self.timestamp_column,
            attribute_cols=self.target_features[0].df_column_name,
            fill=True,
            case_level=False,
            title="Transition duration development",
        )

        # case duration distribution
        fig_distribution = DistributionFigure(
            df=self.df_target,
            attribute_col=self.target_features[0].df_column_name,
            attribute_name="Transition time",
            num_bins=10,
        )

        return self.create_box(
            [
                [metrics_box],
                [fig_transition_duration_development.figure],
                [fig_distribution.figure],
            ]
        )


class OverviewScreenRework(OverviewScreen):
    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        features: List[Feature],
        target_features: List[Feature],
        timestamp_column: str,
        time_aggregation: str,
        # activities: str,
        case_duration_col_name: str,
        num_cases: int,
    ):
        self.df_x = df_x
        self.df_target = df_target
        self.features = features
        self.target_feature = target_features[0]
        self.timestamp_column = timestamp_column
        self.time_aggregation = utils.get_aggregation_df_name(time_aggregation)
        # self.activities = activities
        self.case_duration_col_name = case_duration_col_name
        self.num_cases = num_cases
        self.activity = self._get_activities()
        self._overview_box = self._create_overview_screen()

    @property
    def overview_box(self):
        return self._overview_box

    def _get_activities(self):
        return self.target_feature.attribute.activity

    def _create_overview_screen(self):
        """Create and get the overview screen

        :return:
        """
        # Transitions from source activity to selected target activities. It is the
        # length of the DataFrame without the rows where there is no target activity
        # (Then after the source activity none of the target activities occur.)
        # Get indices where there is a decision to one of the target activities
        col_name = self.target_feature.df_column_name
        idxs_with_target = np.where(self.df_target[col_name] == 1)[0]
        indices_with_rework = idxs_with_target.tolist()

        # Total number of selected cases
        title = "Number of selected cases"
        num_cases_box = SingleValueBox(
            title=title,
            val=self.num_cases,
            unit=None,
            title_color="Black",
            val_color="Blue",
        )

        # Number of cases with at least one rework on the selected activities
        num_cases_with_rework = len(
            set(self.df_target.iloc[indices_with_rework].index.get_level_values(0))
        )
        num_cases_with_activity = len(self.df_target.index.get_level_values(0))

        title = (
            f"Number of cases with rework on at least on of the selected "
            f"activity {self.activity}"
        )
        num_cases_with_rework_box = SingleValueBox(
            title=title,
            val=num_cases_with_rework,
            unit=None,
            title_color="Black",
            val_color="Blue",
        )

        metrics_box = HBox(
            children=[
                num_cases_box.box,
                num_cases_with_rework_box.box,
            ],
            layout=Layout(margin="0px 30px 0px 0px"),
        )

        # Get average case duration
        if len(self.df_target[self.df_target[col_name] == 1].index) == 0:
            avg_case_duration_with_rework = 0
        else:
            # self.activity_case_key
            df_with_target = self.df_x[self.df_target[col_name] == 1]
            df_grouped = df_with_target.groupby(level=0).first()
            avg_case_duration_with_rework = round(
                df_grouped[self.case_duration_col_name].mean(),
                2,
            )
        if len(self.df_target[self.df_target[col_name] == 0].index) == 0:
            avg_case_duration_without_rework = 0
        else:
            # self.activity_case_key
            df_with_target = self.df_x[self.df_target[col_name] == 0]
            df_grouped = df_with_target.groupby(level=0).first()
            avg_case_duration_without_rework = round(
                df_grouped[self.case_duration_col_name].mean(),
                2,
            )

        num_cases_with_rework = self.target_feature.metrics["case_count"]

        # barplot with cases with target activities and metric line plot

        barplot_num_cases_args = [
            {
                "x": [""],
                "y": [num_cases_with_activity],
                "name": "Cases with activity",
                "marker": dict(color="darkblue"),
            },
            {
                "x": [""],
                "y": [num_cases_with_rework],
                "name": "Cases with rework on activity",
                "marker": dict(color="lightblue"),
            },
        ]

        layout_num_cases_args = {
            "xaxis_title": "Rework on activity",
            "yaxis_title": "Cases",
            "title": "Cases with activity and rework on activity",
        }
        barplot_num_cases = GroupedBar(barplot_num_cases_args, **layout_num_cases_args)

        barplot_case_durations_args = [
            {
                "x": [""],
                "y": [avg_case_duration_with_rework],
                "name": "With rework on activity",
                "marker": dict(color="darkblue"),
            },
            {
                "x": [""],
                "y": [avg_case_duration_without_rework],
                "name": "Without rework on activity",
                "marker": dict(color="lightblue"),
            },
        ]

        layout_case_durations_args = {
            "xaxis_title": "Activities",
            "yaxis_title": f"Average case duration ({self.time_aggregation})",
            "title": "Average case durations of cases with/without rework on an "
            "activity",
        }

        barplot_case_duration = GroupedBar(
            barplot_args=barplot_case_durations_args, **layout_case_durations_args
        )
        # development of case duration
        title_rework_development = "Cases with rework on activities"
        df_target_with_case_time = self.df_target
        df_target_with_case_time[self.timestamp_column] = self.df_x[
            self.timestamp_column
        ]
        fig_rework_development = AttributeDevelopmentFigure(
            df=df_target_with_case_time,
            time_col=self.timestamp_column,
            attribute_cols=col_name,
            attribute_names=self.activity,
            fill=False,
            case_level=True,
            case_level_aggregation="max",
            title=title_rework_development,
            data_aggregation=np.sum,
        )

        return self.create_box(
            [
                [metrics_box],
                [barplot_num_cases.figure],
                [barplot_case_duration.figure],
                [fig_rework_development.figure],
            ]
        )
