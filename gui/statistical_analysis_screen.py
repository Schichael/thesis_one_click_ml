import functools
from typing import Union

import plotly.graph_objects as go
from ipywidgets import Box
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

import utils
from feature_processing.attributes import Attribute
from feature_processing.attributes import AttributeDataType
from feature_processing.feature_processor import FeatureProcessor


class StatisticalAnalysisBox:
    def __init__(self, fp: FeatureProcessor):
        """

        :param fp: FeatureProcessor with processed features
        """
        self.fp = fp

    def create_statistical_screen(self) -> VBox:
        """Create and get the screen for the statistical analysis

        :return: box with the screen for the statistical analysis
        """
        statistical_screen_box = VBox()
        title_scrollbox_layout = Layout(margin="5px 0px 0px 0px")
        title_scrollbox_html = (
            '<span style="font-weight:bold;  font-size:16px"> '
            "Attributes with potential effect on case "
            "duration:</span>"
        )
        title_scrollbox = HTML(title_scrollbox_html, layout=title_scrollbox_layout)
        scroll_box_layout = Layout(
            overflow_y="scroll",
            max_height="400px",
            border="3px solid grey",
            padding="3px 3px 3px 3px",
        )
        scroll_box = VBox(layout=scroll_box_layout)
        attr_boxes = []

        # sort attributes by correlation coefficient
        attrs_sorted = sorted(
            self.fp.attributes, key=lambda x: abs(x.correlation), reverse=True
        )

        for attr in attrs_sorted:
            if attr.correlation >= 0.3:
                attr_field = AttributeField(
                    attr, "Case duration", self.fp, statistical_screen_box
                )
                attr_boxes.append(attr_field.attribute_box)
        scroll_box.children = attr_boxes
        statistical_screen_box.children = [title_scrollbox, scroll_box, VBox()]

        return statistical_screen_box


class AttributeField:
    def __init__(
        self,
        attribute: Attribute,
        label: str,
        fp: FeatureProcessor,
        statistical_screen_box: Box,
    ):
        """

        :param attribute: Attribute object containing the attribute information
        :param label: label of the dependent variable to use for visualization
        :param fp: FeatureProcessor with processed features
        :param statistical_screen_box: the box that contains the statistical screen
        """
        self.label = label
        self.attribute = attribute
        self.fp = fp
        self.statistical_screen_box = statistical_screen_box
        self.attribute_name_label = self.create_attribute_label()
        self.metrics_label = self.create_metrics_label()
        self.button = self.create_button()
        self.attribute_box = self.create_attribute_box()

    def create_attribute_box(self):
        """Create the attribute box

        :return: attribute box
        """
        layout_vbox = Layout(
            border="2px solid gray",
            min_height="100px",
            width="auto",
            padding="0px 0px 0px 0px",
            margin="0px 3px 3px 3px",
        )
        attr_box = VBox(
            children=[self.attribute_name_label, self.metrics_label, self.button],
            layout=layout_vbox,
        )
        return attr_box

    def create_attribute_label(self) -> HTML:
        """Create label for the attribute name

        :return: widgets.HTML object with the attribute name
        """
        html_attribute = (
            '<span style="font-weight:bold"> Attribute: '
            + f'<span style="color: Blue">{self.attribute.display_name}</span></span>'
        )
        attribute_label = HTML(html_attribute, layout=Layout(padding="0px 0px 0px 0px"))
        return attribute_label

    def create_metrics_label(self) -> Union[HBox, HTML]:
        """Create label for the attribute metrics

        :return: box with the metrics
        """
        layout_padding = Layout(padding="0px 0px 0px 12px")
        correlation_html = '<span style="font-weight:bold"> Correlation: </span>' + str(
            round(self.attribute.correlation, 2)
        )
        correlation_label = HTML(correlation_html, layout=layout_padding)
        if self.attribute.attribute_data_type == AttributeDataType.NUMERICAL:
            return correlation_label
        else:
            sign = "+" if self.attribute.label_influence > 0 else ""
            case_duration_effect_html = (
                '<span style="font-weight:bold">Effect on '
                + self.label
                + ": </span>"
                + sign
                + str(round(self.attribute.label_influence))
                + "\xa0"
                + self.fp.label.unit
            )

            case_duration_effect_label = HTML(case_duration_effect_html)
            cases_with_attribute_html = (
                '<span style="font-weight:bold">Cases with attribute: </span>'
                + str(self.attribute.cases_with_attribute)
            )

            cases_with_attribute_label = HTML(
                cases_with_attribute_html, layout=layout_padding
            )
            metrics_box = HBox(
                [
                    case_duration_effect_label,
                    cases_with_attribute_label,
                    correlation_label,
                ]
            )
            return metrics_box

    def create_button(self) -> Button:
        """Create button that when clicked lets the user view details of the attribute

        :return: button
        """
        button_layout = Layout(min_height="30px")
        button = Button(description="Details", layout=button_layout)

        def on_button_clicked(b, statistical_screen_box: Box):
            attribute_box = self.gen_attribute_details_box(self.attribute)
            statistical_screen_box.children = statistical_screen_box.children[:-1] + (
                attribute_box,
            )

        partial_button_clicked = functools.partial(
            on_button_clicked,
            statistical_screen_box=self.statistical_screen_box,
        )
        button.on_click(partial_button_clicked)
        return button

    def gen_attribute_details_box(self, attribute: Attribute):
        """Generate box with the attribute details.

        :param attribute: the attribute for which to create the details box
        :return:
        """

        layout_box = Layout(border="3px solid grey", padding="5px 5px 5px 5px")

        title_layout = Layout(margin="15px 0px 0px 0px")
        title_html = (
            '<span style="font-weight:bold;  font-size:16px"> Attribute details:</span>'
        )
        title_label = HTML(title_html, layout=title_layout)

        fig_layout = Layout(margin="15px 0px 0px 0px", width="100%")

        label_attribute = self.attribute_name_label

        if attribute.attribute_data_type == AttributeDataType.CATEGORICAL:

            hbox_metrics = self.gen_avg_metrics_box()
            df_attr = self.fp.df[
                ["caseid", "Case start time", attribute.df_attribute_name]
            ]
            df_attr["starttime"] = (
                df_attr["Case start time"]
                .dt.to_period(
                    " \
                    "
                    "M"
                )
                .astype(str)
            )
            num_cases_all_df = (
                df_attr.groupby("starttime", as_index=False)["caseid"].count().fillna(0)
            )
            num_cases_all_df = num_cases_all_df.rename({"caseid": "All cases"}, axis=1)
            num_cases_attr_true = (
                df_attr[df_attr[attribute.df_attribute_name] == 1]
                .groupby("starttime", as_index=False)["caseid"]
                .count()
                .fillna(0)
            )
            num_cases_attr_true = num_cases_attr_true.rename(
                {"caseid": "Cases with attribute"}, axis=1
            )
            complete_df = utils.join_dfs(
                [num_cases_all_df, num_cases_attr_true], keys=["starttime"] * 2
            ).fillna(0)
            fig = go.Figure(layout_title_text="Attribute development")
            fig.add_trace(
                go.Scatter(
                    x=complete_df["starttime"],
                    y=complete_df["All cases"],
                    fill="tonexty",
                    name="All cases",
                )
            )  # fill down to xaxis
            fig.add_trace(
                go.Scatter(
                    x=complete_df["starttime"],
                    y=complete_df["Cases with attribute"],
                    fill="tozeroy",
                    name="Cases with attribute",
                )
            )  # fill to trace0 y
            fig.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                height=300,
                margin={"l": 10, "r": 10, "t": 40, "b": 10},
            )
            fig_widget = go.FigureWidget(fig)
            fig_box = VBox([fig_widget], layout=fig_layout)
            vbox_details = VBox(
                children=[label_attribute, hbox_metrics, fig_box], layout=layout_box
            )

        else:
            df_attr = self.fp.df[
                ["caseid", "Case start time", attribute.df_attribute_name]
            ]
            df_attr["starttime"] = df_attr["starttime"].dt.to_period("M").astype(str)
            avg_case_duration_over_attribute = (
                df_attr.groupby(attribute.df_attribute_name, as_index=False)[
                    "Case duration"
                ]
                .mean()
                .fillna(0)
            )
            # Attribute effect on label
            fig_effect = go.Figure(
                layout_title_text="Case duration over attribute value"
            )
            fig_effect.add_trace(
                go.Scatter(
                    x=avg_case_duration_over_attribute[attribute.df_attribute_name],
                    y=avg_case_duration_over_attribute["Case duration"],
                    fill="tonexty",
                )
            )
            fig_effect.update_layout(
                title="Effect of attribute on case duration",
                xaxis_title=None,
                yaxis_title=None,
                height=250,
                margin={"l": 5, "r": 10, "t": 40, "b": 10},
            )
            fig_effect_widget = go.FigureWidget(fig_effect)
            fig_effect_box = VBox([fig_effect_widget], layout=fig_layout)

            avg_attribute_over_time_df = (
                df_attr.groupby("starttime", as_index=False)[
                    attribute.df_attribute_name
                ]
                .mean()
                .fillna(0)
            )
            fig_dev = go.Figure(layout_title_text="Attribute value development")
            fig_dev.add_trace(
                go.Scatter(
                    x=avg_attribute_over_time_df["starttime"],
                    y=avg_attribute_over_time_df[attribute.df_attribute_name],
                    fill="tonexty",
                )
            )
            fig_dev.update_layout(
                title="Average attribute value development",
                xaxis_title=None,
                yaxis_title=None,
                height=250,
                margin={"l": 5, "r": 10, "t": 40, "b": 10},
            )
            fig_dev_widget = go.FigureWidget(fig_dev)
            fig_dev_box = VBox([fig_dev_widget], layout=fig_layout)

            vbox_details = VBox(
                children=[label_attribute, fig_effect_box, fig_dev_box],
                layout=layout_box,
            )

        vbox_whole = VBox([title_label, vbox_details])
        return vbox_whole

    def gen_avg_metrics_box(self):
        """Generate box with the average influence of the attribute on dependent
        variable

        :return: box with average metrics
        """
        avg_with_attr = round(
            self.fp.df[self.fp.df[self.attribute.df_attribute_name] == 1][
                self.fp.label.df_attribute_name
            ].mean()
        )
        avg_without_attr = round(
            self.fp.df[self.fp.df[self.attribute.df_attribute_name] != 1][
                self.fp.label.df_attribute_name
            ].mean()
        )

        html_avg_with_attr = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Average '
                    + self.label
                    + " with attribute"
                    + '</span><br><span style="color: Red; font-size:16px; '
                    'text-align: center">'
                    + str(avg_with_attr)
                    + "\xa0"
                    + self.fp.label.unit
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue", margin="0px 10px 0px 0px"
            ),
        )
        html_avg_without_attr = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold; text-align: center"> '
                    "Average "
                    + self.label
                    + " without attribute"
                    + '</span><br><span style="color: Green; font-size:16px; '
                    'text-align: center">'
                    + str(avg_without_attr)
                    + "\xa0"
                    + self.fp.label.unit
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue",
                color="CornflowerBlue",
                margin="0px 0px 0px 10px",
            ),
        )
        hbox_metrics_layout = Layout(margin="5px 0px 0px 0px")
        hbox_metrics = HBox(
            [html_avg_with_attr, html_avg_without_attr], layout=hbox_metrics_layout
        )
        return hbox_metrics
