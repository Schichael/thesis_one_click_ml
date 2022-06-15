import functools
from typing import List
from typing import Union

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import Box
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

from one_click_analysis.gui.figures import AttributeDevelopmentFigure
from one_click_analysis.violation_processing.violation import Violation
from one_click_analysis.violation_processing.violation import ViolationType


class ViolationSelectionScreen:
    def __init__(
        self,
        violations: List[Violation],
        violation_df: pd.DataFrame,
        timestamp_column: str,
        time_aggregation: str,
    ):
        """

        :param violations: List with violations
        :param violation_df: DataFrame with violations
        :param time_aggregation: One of [DAYS, HOURS, MINUTES, SECONDS]
        """
        self.violations = violations
        self.violation_df = violation_df
        self.time_aggregation = time_aggregation
        self.timestamp_column = timestamp_column
        # self.fp = fp
        self.violations_box = None
        self.violations_box_contents = []
        self.violation_selection_box = VBox()

    def create_title_violations_box(self) -> VBox:
        """Create the title of the attributes box.

        :return: HTML widget with the title
        """
        title_attributes_box_layout = Layout(margin="5px 0px 0px 0px")

        title = (
            '<span style="font-weight:bold;  font-size:16px"> '
            "Process model violations</span>"
        )
        title_attributes_html = HTML(title)
        title_box = VBox(children=[title_attributes_html])
        title_box.layout = title_attributes_box_layout
        return title_box

    def create_violations_box_contents(self):
        violation_boxes_list = []
        violation_boxes = []

        # sort features by correlation coefficient
        violations_sorted = sorted(
            self.violations,
            key=lambda x: x.num_cases,
            reverse=True,
        )

        for violation in violations_sorted:
            violation_field = ViolationField(
                violation=violation,
                violation_df=self.violation_df,
                violation_screen_box=self.violation_selection_box,
                time_aggregation=self.time_aggregation,
                timestamp_column=self.timestamp_column,
            )
            violation_boxes.append(violation_field.violation_box)
        violation_boxes_list.append(violation_boxes)
        self.violations_box_contents = violation_boxes_list

    def create_violations_box(self, label_index: int) -> VBox:
        """Create the violations box.

        :return: VBox with the attributes box
        """
        violations_box_layout = Layout(
            overflow="scroll",
            max_height="400px",
            border="3px solid grey",
            padding="3px 3px 3px 3px",
        )
        violations_box = VBox(layout=violations_box_layout)

        violations_box.children = self.violations_box_contents[label_index]
        return violations_box

    def create_violation_selection_screen(
        self,
    ):
        """populate the screen for the statistical analysis.

        :return: box with the screen for the statistical analysis
        """

        title_violations_box = self.create_title_violations_box()
        self.create_violations_box_contents()
        self.violations_box = self.create_violations_box(0)
        self.violation_selection_box.children = [
            title_violations_box,
            self.violations_box,
            VBox(),
        ]


dividers_time_aggregation = {
    "SECONDS": 1,
    "MINUTES": 60,
    "HOURS": 3600,
    "DAYS": 86400,
}


class ViolationField:
    def __init__(
        self,
        violation: Violation,
        violation_df: pd.DataFrame,
        timestamp_column: str,
        violation_screen_box: Box,
        time_aggregation: str,
    ):
        """

        :param feature:
        :param target_feature:
        :param violation_screen_box: the box that contains the violation screen
        :param time_aggregation: one of [DAYS, HOURS, MINUTES, SECONDS]
        """

        self.violation = violation
        self.violation_df = violation_df
        self.timestamp_column = timestamp_column
        self.violation_screen_box = violation_screen_box
        self.time_aggregation = time_aggregation
        self.violation_name_label = self.create_violation_label()
        self.metrics_label = self.create_metrics_label()
        self.button = self.create_button()
        self.violation_box = self.create_violation_box()

    def _get_divider_case_duration(self, time_aggregation):
        """Get divider for time aggregation

        :param time_aggregation: string with the time aggregation
        :return:
        """

    def create_violation_box(self):
        """Create the feature box

        :return: details box
        """
        layout_vbox = Layout(
            border="2px solid gray",
            min_height="105px",
            width="auto",
            padding="0px 0px 0px 0px",
            margin="0px 3px 3px 3px",
        )
        violation_box = VBox(
            children=[self.violation_name_label, self.metrics_label, self.button],
            layout=layout_vbox,
        )
        return violation_box

    def create_violation_label(self) -> HTML:
        """Create label for the violation name

        :return: widgets.HTML object with the feature name
        """
        if self.violation.violation_type != ViolationType.INCOMPLETE:
            violation_str = self.violation.violation_readable
        else:
            violation_str = self.violation.violation_readable + " Case"
        html_feature = (
            '<span style="font-weight:bold"> Violation: '
            + f'<span style="color: Blue">{violation_str}</span></span>'
        )
        feature_label = HTML(html_feature, layout=Layout(padding="0px 0px 0px 0px"))
        return feature_label

    def create_metrics_label(self) -> Union[HBox, HTML]:
        """Create label for the attribute metrics

        :return: box with the metrics
        """
        layout_padding = Layout(padding="0px 0px 0px 12px")
        num_cases_html = (
            '<span style="font-weight:bold"> Cases with violation: '
            "</span>" + str(round(self.violation.num_cases))
        )
        num_cases_label = HTML(num_cases_html)
        effect_on_case_duration_raw = self.violation.metrics["effect_on_case_duration"]
        effect_on_case_duration = (
            effect_on_case_duration_raw
            / dividers_time_aggregation[self.time_aggregation]
        )
        sign_case_duration = "+" if effect_on_case_duration_raw > 0 else ""
        effect_on_case_duration_html = (
            '<span style="font-weight:bold">Effect on '
            + "case duration: </span>"
            + sign_case_duration
            + str(round(effect_on_case_duration, 1))
            + "\xa0"
            + self.time_aggregation.lower()
        )
        effect_on_case_duration_label = HTML(
            effect_on_case_duration_html, layout=layout_padding
        )

        effect_on_event_count = self.violation.metrics["effect_on_event_count"]
        sign_event_count = "+" if effect_on_event_count > 0 else ""
        effect_on_event_count_html = (
            '<span style="font-weight:bold">Effect on ' + "event count: "
            "</span>" + sign_event_count + str(round(effect_on_event_count, 1))
        )
        effect_on_event_count_label = HTML(
            effect_on_event_count_html, layout=layout_padding
        )

        metrics_box = HBox(
            [
                num_cases_label,
                effect_on_case_duration_label,
                effect_on_event_count_label,
            ]
        )

        return metrics_box

    def create_button(self) -> Button:
        """Create button that when clicked lets the user view details of the attribute

        :return: button
        """
        button_layout = Layout(min_height="30px")
        button = Button(description="Details", layout=button_layout)

        def on_button_clicked(b, violation_screen_box: Box):
            attribute_box = self.gen_violation_details_box()
            violation_screen_box.children = violation_screen_box.children[:-1] + (
                attribute_box,
            )

        partial_button_clicked = functools.partial(
            on_button_clicked,
            violation_screen_box=self.violation_screen_box,
        )
        button.on_click(partial_button_clicked)
        return button

    def gen_violation_details_box(self):
        """Generate box with the attribute details.

        :return:
        """

        layout_box = Layout(border="3px solid grey", padding="5px 5px 5px 5px")

        title_layout = Layout(margin="15px 0px 0px 0px")
        title_html = (
            '<span style="font-weight:bold;  font-size:16px"> Attribute details:</span>'
        )
        title_label = HTML(title_html, layout=title_layout)

        fig_layout = Layout(margin="15px 0px 0px 0px", width="100%")

        label_attribute = self.violation_name_label

        vbox_metrics = self.gen_avg_metrics_box()

        df_current_violation = self.violation_df[
            ["case", self.timestamp_column, "violation"]
        ]
        cases_violation = self.violation_df[
            self.violation_df["violation"] == self.violation.violation_readable
        ]["case"].unique()

        cases_with_violation_str = f"Cases with violation"
        cases_all_str = f"Cases (all)"
        df_current_violation[cases_all_str] = 1
        df_current_violation[cases_with_violation_str] = 0
        df_current_violation.loc[
            df_current_violation["case"].isin(cases_violation) == 1,
            f"Cases with violation",
        ] = 1
        fig_attribute_development = AttributeDevelopmentFigure(
            df=df_current_violation,
            time_col=self.timestamp_column,
            attribute_cols=[cases_all_str, cases_with_violation_str],
            time_aggregation="M",
            data_aggregation="sum",
            case_level=True,
            case_level_aggregation="max",
            fill=True,
        )

        fig_widget = go.FigureWidget(fig_attribute_development.figure)
        fig_box = VBox([fig_widget], layout=fig_layout)
        vbox_details = VBox(
            children=[label_attribute, vbox_metrics, fig_box], layout=layout_box
        )

        vbox_whole = VBox([title_label, vbox_details])
        return vbox_whole

    def gen_avg_metrics_box(self) -> VBox:
        """Generate box with the average influence of the attribute on dependent
        variable

        :return: box with average metrics
        """

        avg_case_duration_with_violation = self.violation.metrics[
            "avg_case_duration_with_violation"
        ]
        avg_case_duration_with_violation = round(
            avg_case_duration_with_violation
            / dividers_time_aggregation[self.time_aggregation],
            1,
        )

        avg_case_duration_without_violation = self.violation.metrics[
            "avg_case_duration_without_violation"
        ]
        avg_case_duration_without_violation = round(
            avg_case_duration_without_violation
            / dividers_time_aggregation[self.time_aggregation],
            1,
        )

        html_avg_case_duration_with_violation = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Average '
                    + "duration of cases with violation"
                    + '</span><br><span style="color: Blue; font-size:16px; '
                    'text-align: center">'
                    + str(avg_case_duration_with_violation)
                    + "\xa0"
                    + self.time_aggregation.lower()
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue", margin="0px 10px 0px 0px"
            ),
        )

        html_avg_case_duration_without_violation = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Average '
                    + "Average duration of "
                    "cases "
                    "without "
                    "violation"
                    + '</span><br><span style="color: Blue; font-size:16px; '
                    'text-align: center">'
                    + str(avg_case_duration_without_violation)
                    + "\xa0"
                    + self.time_aggregation.lower()
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue", margin="0px 10px 0px 0px"
            ),
        )

        avg_events_with_violation = round(
            self.violation.metrics["avg_num_events_with_violation"]
        )

        avg_events_without_violation = round(
            self.violation.metrics["avg_num_events_without_violation"]
        )

        html_num_events_with_violation = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Average '
                    + "Average event count "
                    "of cases "
                    "with "
                    "violation"
                    + '</span><br><span style="color: Blue; font-size:16px; '
                    'text-align: center">'
                    + str(avg_events_with_violation)
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue", margin="0px 10px 0px 0px"
            ),
        )

        html_num_events_without_violation = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Average '
                    + "Average event count of cases "
                    "without "
                    "violation"
                    + '</span><br><span style="color: Blue; font-size:16px; '
                    "text-align: "
                    'center">' + str(avg_events_without_violation) + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue", margin="0px 10px 0px 0px"
            ),
        )

        hbox_metrics_layout = Layout(margin="5px 0px 0px 0px")
        hbox_metrics_case_duration = HBox(
            [
                html_avg_case_duration_with_violation,
                html_avg_case_duration_without_violation,
            ],
            layout=hbox_metrics_layout,
        )
        hbox_metrics_event_count = HBox(
            [html_num_events_with_violation, html_num_events_without_violation],
            layout=hbox_metrics_layout,
        )

        vbox_metrics = VBox(
            [hbox_metrics_case_duration, hbox_metrics_event_count],
            layout=hbox_metrics_layout,
        )

        return vbox_metrics
