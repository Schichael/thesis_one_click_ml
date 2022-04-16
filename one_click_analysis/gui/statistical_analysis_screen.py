import functools
from typing import List
from typing import Union

import numpy as np
import plotly.graph_objects as go
from ipywidgets import Box
from ipywidgets import Button
from ipywidgets import Dropdown
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

from one_click_analysis.feature_processing import attributes
from one_click_analysis.feature_processing.feature_processor import FeatureProcessor
from one_click_analysis.gui.figures import AttributeDevelopmentFigure


class StatisticalAnalysisScreen:
    def __init__(
        self,
        fp: FeatureProcessor,
        th: float,
        selected_attributes: List[attributes.MinorAttribute],
        selected_activity_table_cols: List[str],
        selected_case_table_cols: List[str],
    ):
        """

        :param fp: FeatureProcessor with processed features
        :param th: threshold for correlation coefficient
        :param selected_attributes:
        """
        self.fp = fp
        self.th = th
        self.selected_attributes = selected_attributes
        self.selected_activity_table_cols = selected_activity_table_cols
        self.selected_case_table_cols = selected_case_table_cols
        self.attributes_box = None
        self.attributes_box_contents = []
        self.statistical_analysis_box = VBox()

    def update_attr_selection(
        self,
        selected_attributes,
        selected_activity_table_cols,
        selected_case_table_cols,
    ):
        """Define behaviour when the attribute selection is updated. Here, the screen is
        simply constructed again with the new attributes.

        :param selected_attributes: list with the selected MinorAttributes
        :param selected_activity_table_cols:
        :param selected_case_table_cols:
        :return:
        """
        self.selected_attributes = selected_attributes
        self.selected_activity_table_cols = selected_activity_table_cols
        self.selected_case_table_cols = selected_case_table_cols
        self.create_statistical_screen()

    def create_title_attributes_box(self) -> VBox:
        """Create the title of the attributes box.

        :return: HTML widget with the title
        """
        title_attributes_box_layout = Layout(margin="5px 0px 0px 0px")
        if len(self.fp.labels) == 1:
            label_str = self.fp.labels[0].display_name
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
            print(change)
            if change.new != change.old:
                self.attributes_box.children = self.attributes_box_contents[change.new]

        if len(self.fp.labels) == 1:
            title_box = VBox(children=[title_attributes_html])
        else:
            dropdpwn_options = [
                (label.display_name, i) for i, label in enumerate(self.fp.labels)
            ]
            dropdown = Dropdown(options=dropdpwn_options, value=0)
            dropdown.observe(drop_down_on_change, "value")
            title_box = VBox(children=[title_attributes_html, dropdown])

        title_box.layout = title_attributes_box_layout

        return title_box

    def create_attributes_box_contents(self):
        attr_boxes_list = []
        for label_index in range(len(self.fp.labels)):
            attr_boxes = []

            # remove nans
            attr_not_nan = [
                i
                for i in self.fp.attributes
                if not np.isnan(i.correlation[label_index])
            ]

            # sort attributes by correlation coefficient
            attrs_sorted = sorted(
                attr_not_nan,
                key=lambda x: abs(x.correlation[label_index]),
                reverse=True,
            )

            selected_attr_types = tuple([type(i) for i in self.selected_attributes])
            for attr in attrs_sorted:
                if not isinstance(attr.minor_attribute_type, selected_attr_types):
                    continue
                elif isinstance(
                    attr.minor_attribute_type,
                    attributes.ActivityTableColumnMinorAttribute,
                ):
                    if attr.column_name not in self.selected_activity_table_cols:
                        continue
                elif isinstance(
                    attr.minor_attribute_type, attributes.CaseTableColumnMinorAttribute
                ):
                    if attr.column_name not in self.selected_case_table_cols:
                        continue
                if (attr.correlation[label_index] >= self.th) or (
                    attr.attribute_data_type == attributes.AttributeDataType.NUMERICAL
                    and abs(attr.correlation[label_index]) >= self.th
                ):
                    attr_field = AttributeField(
                        attr,
                        self.fp.labels[label_index].display_name,
                        self.fp,
                        label_index,
                        self.statistical_analysis_box,
                    )
                    attr_boxes.append(attr_field.attribute_box)
            attr_boxes_list.append(attr_boxes)
        self.attributes_box_contents = attr_boxes_list

    def create_attributes_box(self, label_index: int) -> VBox:
        """Create the attributes box.

        :return: VBox with the attributes box
        """
        attributes_box_layout = Layout(
            overflow_y="scroll",
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
        self.create_attributes_box_contents()
        self.attributes_box = self.create_attributes_box(0)
        self.statistical_analysis_box.children = [
            title_attributes_box,
            self.attributes_box,
            VBox(),
        ]


class AttributeField:
    def __init__(
        self,
        attribute: attributes.Attribute,
        label: str,
        fp: FeatureProcessor,
        label_index: int,
        statistical_screen_box: Box,
    ):
        """

        :param attribute: Attribute object containing the attribute information
        :param label: label of the dependent variable to use for visualization
        :param fp: FeatureProcessor with processed features
        :param label_index: index of the label in the FeatureProcessor
        :param statistical_screen_box: the box that contains the statistical screen
        """
        self.label = label
        self.attribute = attribute
        self.fp = fp
        self.label_index = label_index
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
            min_height="105px",
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
            round(self.attribute.correlation[self.label_index], 2)
        )
        correlation_label = HTML(correlation_html)
        if self.attribute.attribute_data_type == attributes.AttributeDataType.NUMERICAL:
            return correlation_label
        else:
            sign = "+" if self.attribute.label_influence[self.label_index] > 0 else ""
            effect_on_label_html = (
                '<span style="font-weight:bold">Effect on '
                + self.label
                + ": </span>"
                + sign
                + str(round(self.attribute.label_influence[self.label_index], 2))
                + "\xa0"
                + self.fp.labels[self.label_index].unit
            )

            case_duration_effect_label = HTML(effect_on_label_html)
            cases_with_attribute_html = (
                '<span style="font-weight:bold">Cases with attribute: </span>'
                + str(self.attribute.cases_with_attribute)
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

    def gen_attribute_details_box(self, attribute: attributes.Attribute):
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

        if attribute.attribute_data_type == attributes.AttributeDataType.CATEGORICAL:

            hbox_metrics = self.gen_avg_metrics_box()
            df_attr = self.fp.df[
                ["caseid", "Case start time", attribute.df_attribute_name]
            ]

            df_attr["All Cases"] = 1
            df_attr["Cases with attribute"] = 0
            df_attr.loc[
                df_attr[attribute.df_attribute_name] == 1, "Cases with " "attribute"
            ] = 1
            fig_attribute_development = AttributeDevelopmentFigure(
                df=df_attr,
                time_col="Case start time",
                attribute_cols=["All Cases", "Cases with attribute"],
                time_aggregation="M",
                data_aggregation="sum",
                fill=True,
            )

            fig_widget = go.FigureWidget(fig_attribute_development.figure)
            fig_box = VBox([fig_widget], layout=fig_layout)
            vbox_details = VBox(
                children=[label_attribute, hbox_metrics, fig_box], layout=layout_box
            )

        else:
            df_attr = self.fp.df[
                [
                    "caseid",
                    "Case start time",
                    attribute.df_attribute_name,
                    self.fp.labels[self.label_index].df_attribute_name,
                ]
            ]
            # df_attr["starttime"] = (
            #    df_attr["Case start time"].dt.to_period("M").astype(str)
            # )

            avg_case_duration_over_attribute = (
                df_attr.groupby(attribute.df_attribute_name, as_index=False)[
                    self.fp.labels[self.label_index].df_attribute_name
                ]
                .mean()
                .fillna(0)
            )
            fig_effect = go.Figure(
                layout_title_text=self.fp.labels[self.label_index].display_name
                + " over attribute value"
            )
            # Attribute effect on label
            fig_effect.add_trace(
                go.Scatter(
                    x=avg_case_duration_over_attribute[attribute.df_attribute_name],
                    y=avg_case_duration_over_attribute[
                        self.fp.labels[self.label_index].df_attribute_name
                    ],
                    fill="tonexty",
                )
            )
            fig_effect.update_layout(
                title="Effect of attribute on "
                + self.fp.labels[self.label_index].display_name,
                xaxis_title=None,
                yaxis_title=None,
                height=250,
                margin={"l": 5, "r": 10, "t": 40, "b": 10},
            )
            fig_effect_widget = go.FigureWidget(fig_effect)
            fig_effect_box = VBox([fig_effect_widget], layout=fig_layout)

            attr_dev_fig = AttributeDevelopmentFigure(
                df=self.fp.df,
                time_col="Case start time",
                attribute_cols=attribute.df_attribute_name,
                attribute_names=attribute.display_name,
                time_aggregation="M",
                data_aggregation="mean",
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
            self.fp.df[self.fp.df[self.attribute.df_attribute_name] == 1][
                self.fp.labels[self.label_index].df_attribute_name
            ].mean(),
            2,
        )
        avg_without_attr = round(
            self.fp.df[self.fp.df[self.attribute.df_attribute_name] != 1][
                self.fp.labels[self.label_index].df_attribute_name
            ].mean(),
            2,
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
                    + self.fp.labels[self.label_index].unit
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
                    + self.fp.labels[self.label_index].unit
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
