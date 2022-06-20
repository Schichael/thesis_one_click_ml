import functools
from typing import List
from typing import Union

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import Box
from ipywidgets import Button
from ipywidgets import Dropdown
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDataType,
)
from one_click_analysis.feature_processing.attributes.feature import Feature
from one_click_analysis.gui.figures import AttributeDevelopmentFigure


class StatisticalAnalysisScreen:
    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        features: List[Feature],
        target_features: List[Feature],
        timestamp_column: str,
        datapoint_str: str,
        th: float,
    ):
        """
        :param datapoint_str: string of the name of a datapoint. E.g. 'Cases' or
        'Total transitions'
        :param th: threshold for correlation coefficient
        """
        self.df_x = df_x
        self.df_target = df_target
        self.features = features
        self.target_features = target_features
        self.datapoint_str = datapoint_str
        self.timestamp_column = timestamp_column
        # self.fp = fp
        self.th = th
        self.attributes_box = None
        self.attributes_box_contents = []
        self.statistical_analysis_box = VBox()

    def update_attr_selection(self, features: List[Feature]):
        """Define behaviour when the attribute selection is updated. Here, the screen is
        simply constructed again with the new attributes.

        :param selected_attributes: list with the selected MinorAttributes
        :param selected_activity_table_cols:
        :param selected_case_table_cols:
        :return:
        """
        self.features = features
        self.create_statistical_screen()

    def create_title_attributes_box(self) -> VBox:
        """Create the title of the attributes box.

        :return: HTML widget with the title
        """
        title_attributes_box_layout = Layout(margin="5px 0px 0px 0px")
        if len(self.target_features) == 1:
            label_str = self.target_features[0].df_column_name
            title_attributes_html_str = (
                '<span style="font-weight:bold;  font-size:16px"> '
                "Attributes with potential effect on " + label_str + ":</span>"
            )
        else:
            title_attributes_html_str = (
                '<span style="font-weight:bold;  font-size:16px"> '
                "Attributes with potential effect on </span>"
            )

        title_attributes_html = HTML(title_attributes_html_str)

        def drop_down_on_change(change):
            if change.new != change.old:
                self.attributes_box.children = self.attributes_box_contents[change.new]

        if len(self.target_features) == 1:
            title_box = VBox(children=[title_attributes_html])
        else:
            dropdpwn_options = [
                (label.df_column_name, i)
                for i, label in enumerate(self.target_features)
            ]
            dropdpwn_options_sorted = sorted(dropdpwn_options)
            dropdown = Dropdown(options=dropdpwn_options_sorted, value=0)
            dropdown.observe(drop_down_on_change, "value")
            title_box = VBox(children=[title_attributes_html, dropdown])

        title_box.layout = title_attributes_box_layout

        return title_box

    def create_features_box_contents(self):
        feature_boxes_list = []
        for target_feature in self.target_features:
            target_name = target_feature.df_column_name
            feature_boxes = []
            if len(self.df_target.index) < 3:
                return
            # remove nans
            features_not_nan = [
                i
                for i in self.features
                if not pd.isnull(i.metrics["correlations"][target_name])
            ]

            # sort features by correlation coefficient
            features_sorted = sorted(
                features_not_nan,
                key=lambda x: abs(x.metrics["correlations"][target_name]),
                reverse=True,
            )

            for feature in features_sorted:
                corr = feature.metrics["correlations"][target_name]
                if (corr >= self.th) or (
                    feature.datatype == AttributeDataType.NUMERICAL
                    and abs(corr) >= self.th
                ):

                    attr_field = FeatureField(
                        feature=feature,
                        target_feature=target_feature,
                        df_x=self.df_x,
                        df_target=self.df_target,
                        timestamp_column=self.timestamp_column,
                        datapoint_str=self.datapoint_str,
                        statistical_screen_box=self.statistical_analysis_box,
                    )
                    feature_boxes.append(attr_field.feature_box)
            feature_boxes_list.append(feature_boxes)
        self.attributes_box_contents = feature_boxes_list

    def create_attributes_box(self, label_index: int) -> VBox:
        """Create the attributes box.

        :return: VBox with the attributes box
        """
        attributes_box_layout = Layout(
            overflow="scroll",
            max_height="400px",
            border="3px solid grey",
            padding="3px 3px 3px 3px",
        )
        attributes_box = VBox(layout=attributes_box_layout)

        attributes_box.children = self.attributes_box_contents[label_index]
        return attributes_box

    def create_statistical_screen(
        self,
    ):
        """populate the screen for the statistical analysis.

        :return: box with the screen for the statistical analysis
        """

        title_attributes_box = self.create_title_attributes_box()
        self.create_features_box_contents()
        self.attributes_box = self.create_attributes_box(0)
        self.statistical_analysis_box.children = [
            title_attributes_box,
            self.attributes_box,
            VBox(),
        ]


class FeatureField:
    def __init__(
        self,
        feature: Feature,
        target_feature: Feature,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        timestamp_column: str,
        datapoint_str: str,
        statistical_screen_box: Box,
    ):
        """

        :param feature:
        :param target_feature:
        :param datapoint_str: string of the name of a datapoint. E.g. 'Cases' or
        'Total transitions'
        :param statistical_screen_box: the box that contains the statistical screen
        """

        self.feature = feature
        self.target_feature = target_feature
        self.df_x = df_x
        self.df_target = df_target
        self.timestamp_column = timestamp_column
        self.datapoint_str = datapoint_str
        self.statistical_screen_box = statistical_screen_box
        self.feature_name_label = self.create_feature_label()
        self.metrics_label = self.create_metrics_label()
        self.button = self.create_button()
        self.feature_box = self.create_feature_box()

    def create_feature_box(self):
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
        feature_box = VBox(
            children=[self.feature_name_label, self.metrics_label, self.button],
            layout=layout_vbox,
        )
        return feature_box

    def create_feature_label(self) -> HTML:
        """Create label for the feature name

        :return: widgets.HTML object with the feature name
        """
        html_feature = (
            '<span style="font-weight:bold"> Attribute: '
            + f'<span style="color: Blue">{self.feature.df_column_name}</span></span>'
        )
        feature_label = HTML(html_feature, layout=Layout(padding="0px 0px 0px 0px"))
        return feature_label

    def create_metrics_label(self) -> Union[HBox, HTML]:
        """Create label for the attribute metrics

        :return: box with the metrics
        """
        target_name = self.target_feature.df_column_name
        corr = self.feature.metrics["correlations"][target_name]

        layout_padding = Layout(padding="0px 0px 0px 12px")
        correlation_html = '<span style="font-weight:bold"> Correlation: </span>' + str(
            round(corr, 2)
        )
        correlation_label = HTML(correlation_html)
        if self.feature.datatype == AttributeDataType.NUMERICAL:
            return correlation_label
        else:
            target_influence = self.feature.metrics["target_influence"][target_name]
            sign = "+" if target_influence > 0 else ""
            effect_on_label_html = (
                '<span style="font-weight:bold">Effect on '
                + self.target_feature.df_column_name
                + ": </span>"
                + sign
                + str(round(target_influence, 2))
                + "\xa0"
                + self.feature.unit
            )

            case_duration_effect_label = HTML(effect_on_label_html)
            cases_with_attribute_html = (
                f'<span style="font-weight:bold">{self.datapoint_str} with '
                f"attribute: </span>" + str(self.feature.metrics["case_count"])
            )

            cases_with_attribute_label = HTML(
                cases_with_attribute_html, layout=layout_padding
            )
            correlation_label.layout = layout_padding
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
            attribute_box = self.gen_feature_details_box()
            statistical_screen_box.children = statistical_screen_box.children[:-1] + (
                attribute_box,
            )

        partial_button_clicked = functools.partial(
            on_button_clicked,
            statistical_screen_box=self.statistical_screen_box,
        )
        button.on_click(partial_button_clicked)
        return button

    def gen_feature_details_box(self):
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

        label_attribute = self.feature_name_label

        if self.feature.datatype == AttributeDataType.CATEGORICAL:
            hbox_metrics = self.gen_avg_metrics_box()

            df_attr = self.df_x[[self.timestamp_column, self.feature.df_column_name]]

            datapoint_with_attr_str = f"{self.datapoint_str} with attribute"
            datapoint_all_str = f"{self.datapoint_str} (all)"
            df_attr[datapoint_all_str] = 1
            df_attr[datapoint_with_attr_str] = 0
            df_attr.loc[
                df_attr[self.feature.df_column_name] == 1,
                f"{self.datapoint_str} with attribute",
            ] = 1
            fig_attribute_development = AttributeDevelopmentFigure(
                df=df_attr,
                time_col=self.timestamp_column,
                attribute_cols=[datapoint_all_str, datapoint_with_attr_str],
                time_aggregation="M",
                data_aggregation="sum",
                case_level=True,
                case_level_aggregation="max",
                fill=True,
            )

            fig_widget = go.FigureWidget(fig_attribute_development.figure)
            fig_box = VBox([fig_widget], layout=fig_layout)
            vbox_details = VBox(
                children=[label_attribute, hbox_metrics, fig_box], layout=layout_box
            )

        else:
            df_attr = self.df_x[
                [
                    self.timestamp_column,
                    self.feature.df_column_name,
                ]
            ]
            df_attr[self.target_feature.df_column_name] = self.df_target[
                self.target_feature.df_column_name
            ]

            avg_target_over_attribute = (
                df_attr.groupby(self.feature.df_column_name, as_index=False)[
                    self.target_feature.df_column_name
                ]
                .mean()
                .fillna(0)
            )
            fig_effect = go.Figure(
                layout_title_text=self.target_feature.df_column_name
                + " over attribute value"
            )
            # Attribute effect on label
            fig_effect.add_trace(
                go.Scatter(
                    x=avg_target_over_attribute[self.feature.df_column_name],
                    y=avg_target_over_attribute[self.target_feature.df_column_name],
                    fill="tonexty",
                )
            )
            fig_effect.update_layout(
                title=self.target_feature.df_column_name + " over attribute value",
                xaxis_title=None,
                yaxis_title=None,
                height=250,
                margin={"l": 5, "r": 10, "t": 40, "b": 10},
            )
            fig_effect_widget = go.FigureWidget(fig_effect)
            fig_effect_box = VBox([fig_effect_widget], layout=fig_layout)

            attr_dev_fig = AttributeDevelopmentFigure(
                df=self.df_x,
                time_col=self.timestamp_column,
                attribute_cols=self.feature.df_column_name,
                attribute_names=self.feature.df_column_name,
                time_aggregation="M",
                data_aggregation="mean",
                case_level=False,
                fill=True,
                title="Attribute value development",
            )

            vbox_details = VBox(
                children=[label_attribute, fig_effect_box, attr_dev_fig.figure],
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
            self.df_target[self.df_x[self.feature.df_column_name] == 1][
                self.target_feature.df_column_name
            ].mean(),
            2,
        )
        avg_without_attr = round(
            self.df_target[self.df_x[self.feature.df_column_name] != 1][
                self.target_feature.df_column_name
            ].mean(),
            2,
        )

        html_avg_with_attr = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Average '
                    + self.target_feature.df_column_name
                    + " with attribute"
                    + '</span><br><span style="color: Blue; font-size:16px; '
                    'text-align: center">'
                    + str(avg_with_attr)
                    + "\xa0"
                    + self.target_feature.unit
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
                    + self.target_feature.df_column_name
                    + " without attribute"
                    + '</span><br><span style="color: Blue; font-size:16px; '
                    'text-align: center">'
                    + str(avg_without_attr)
                    + "\xa0"
                    + self.target_feature.unit
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue",
                margin="0px 0px 0px 10px",
            ),
        )
        hbox_metrics_layout = Layout(margin="5px 0px 0px 0px")
        hbox_metrics = HBox(
            [html_avg_with_attr, html_avg_without_attr], layout=hbox_metrics_layout
        )
        return hbox_metrics
