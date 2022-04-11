import functools
from typing import Callable
from typing import List
from typing import Optional

import ipysheet
import ipywidgets as widgets
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Box
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Label
from ipywidgets import Layout
from ipywidgets import VBox
from scipy import stats

from one_click_analysis.decision_rules.decision_rule_miner import DecisionRuleMiner
from one_click_analysis.errors import DecisionRuleNotValidLabelTypesError
from one_click_analysis.errors import MaximumValueReachedError
from one_click_analysis.errors import MinimumValueReachedError
from one_click_analysis.feature_processing import attributes
from one_click_analysis.feature_processing.attributes import AttributeDataType
from one_click_analysis.feature_processing.feature_processor import FeatureProcessor


class DecisionRulesScreen:
    def __init__(
        self,
        fp: FeatureProcessor,
        selected_attributes: List[attributes.MinorAttribute],
        selected_activity_table_cols: List[str],
        selected_case_table_cols: List[str],
    ):
        """
        :param fp: FeatureProcessor with processed features
        """
        self.fp = fp
        self.selected_attributes = selected_attributes
        self.selected_activity_table_cols = selected_activity_table_cols
        self.selected_case_table_cols = selected_case_table_cols
        self.attr_names = self.create_attribute_names()
        # Stores the dr miners. If the label is numerical, the miner will be stored
        # with key 0
        self.dr_miners = {}
        self.decision_rules = {}
        self.decision_rule_box = VBox()
        self.run_buttons = {}
        self.buttons_elaborate_rules = {}
        self.buttons_simplify_rules = {}
        # the currently selected threshold
        self.current_threshold_numerical = None
        self.is_numerical = self._set_numerical()
        self.value_selection = self._create_ValueSelection()
        self.parent_rule_box = self._init_rules_parent_box()

    def _set_numerical(self):
        """Set the is_numerical member variable

        :return: True if label is numerical and False if labels are all categorical
        """
        datatypes = [label.attribute_data_type for label in self.fp.labels]
        if len(datatypes) == 1:
            if datatypes[0] == attributes.AttributeDataType.NUMERICAL:
                return True
            else:
                return False
        elif (
            len(self.fp.labels) > 1
            and attributes.AttributeDataType.NUMERICAL not in datatypes
        ):
            return False
        else:
            raise DecisionRuleNotValidLabelTypesError(datatypes)

    def _create_ValueSelection(self):
        """Create the ValueSelection object.

        :return: if self.is_numerical is true, then the ValueSelection object,
        else None
        """
        if self.is_numerical:
            value_selection = ValueSelection(self.fp, self.on_button_run_clicked)
            self.run_buttons[0] = value_selection.button_run
            return value_selection
        else:
            return None

    def update_attr_selection(
        self,
        selected_attributes,
        selected_activity_table_cols,
        selected_case_table_cols,
    ):
        """Define behaviour when the attribute selection is updated. Here, the screen is
        simply constructed again with the new attributes.

        :param selected_attributes:
        :param selected_activity_table_cols:
        :param selected_case_table_cols:
        :return:
        """
        self.selected_attributes = selected_attributes
        self.selected_activity_table_cols = selected_activity_table_cols
        self.selected_case_table_cols = selected_case_table_cols
        self.rule_box = HBox()
        self.dr_miners = {}
        self.current_threshold_numerical = None
        self.attr_names = self.create_attribute_names()

        self.create_decision_rule_screen()

    def create_attribute_names(self) -> List[str]:
        """Create DataFrame with the selected attribute values and activity and
        case columns and the dependent variable

        :return: Dataframe with the attributes and the dependent variable
        """
        selected_attr_types = tuple([type(i) for i in self.selected_attributes])
        attr_col_names = []
        for attr in self.fp.attributes:
            if not isinstance(
                attr.minor_attribute_type, attributes.ActivityTableColumnMinorAttribute
            ) and not isinstance(
                attr.minor_attribute_type, attributes.CaseTableColumnMinorAttribute
            ):
                if isinstance(attr.minor_attribute_type, selected_attr_types):
                    attr_col_names.append(attr.df_attribute_name)
            elif isinstance(
                attr.minor_attribute_type, attributes.ActivityTableColumnMinorAttribute
            ):
                if attr.column_name in self.selected_activity_table_cols:
                    attr_col_names.append(attr.df_attribute_name)
            elif isinstance(
                attr.minor_attribute_type, attributes.CaseTableColumnMinorAttribute
            ):
                if attr.column_name in self.selected_case_table_cols:
                    attr_col_names.append(attr.df_attribute_name)
        return attr_col_names

    def create_decision_rule_screen(self):
        """Create and get the decision rule screen, i.e. the box that contains the
        selection box and the rule box

        :return: box with the decision rule screen
        """
        self.decision_rule_box.children = [
            self.value_selection.selection_box,
            self.parent_rule_box,
        ]

    def on_button_run_clicked(self, b, label_index: int):
        if self.is_numerical:
            threshold = self.value_selection.get_selected_threshold()
            # Do not run the miner twice for the same threshold value
            if threshold == self.current_threshold_numerical:
                return
            self.current_threshold_numerical = threshold
            pos_class = None
        else:
            threshold = None
            pos_class = 1

        self.run_buttons[label_index].disabled = True
        self.dr_miners[label_index] = DecisionRuleMiner(
            self.fp.df,
            self.fp.labels[0].df_attribute_name,
            self.attr_names,
            pos_class=pos_class,
            threshold=threshold,
        )
        self.run_decision_miner(label_index)
        rule_box = self.create_rule_box(label_index)
        children = self.parent_rule_box.children
        children_first = children[:label_index]
        children_last = children[label_index + 1 :]

        children = list(children_first) + [rule_box] + list(children_last)
        self.parent_rule_box.children = children
        self.run_buttons[label_index].disabled = False

    def get_percentile(self, series: pd.Series, val: float):
        """Get the percentile of a value in a series

        :param series: series with numerical values
        :param val: value for which to compute the percentile
        :return: the percentile of the value in the series
        """
        return round(100 - stats.percentileofscore(series, val))

    def _gen_rule_caption(self, label_str: str):
        layout = Layout(margin="0px 0px 10px 0px")
        if self.is_numerical:
            html_rule_caption = HTML(
                '<span style="font-weight:bold; font-size: 16px">'
                + "Rule for "
                + label_str
                + ">=\xa0"
                + str(self.current_threshold_numerical)
                + ":</span>",
                layout=layout,
            )
        else:
            html_rule_caption = HTML(
                '<span style="font-weight:bold; font-size: 16px">'
                + "Rule for "
                + label_str
                + "</span>",
                layout=layout,
            )
        return html_rule_caption

    def _init_rules_parent_box(self):
        layout_rule_box = Layout(
            margin="20px 0px 0px 0px", border="3px groove lightblue"
        )

        labels = [label.display_name for label in self.fp.labels]

        rule_boxes = []
        if self.is_numerical:
            rule_box = HBox(children=[self._gen_rule_caption(labels[0])])
            rule_boxes.append(rule_box)
        else:
            for label_index, label in enumerate(labels):

                button_run = Button(description="Mine rules!")

                button_clicked_partial = functools.partial(
                    self.on_button_run_clicked, label_index=label_index
                )
                button_run.on_click(button_clicked_partial)
                self.run_buttons[label_index] = button_run
                vbox_run_button_layout = Layout(
                    flex="1", justify_content="flex-end", align_items="center"
                )
                vbox_run_button = VBox([button_run], layout=vbox_run_button_layout)
                rule_box = VBox(
                    children=[self._gen_rule_caption(label), vbox_run_button]
                )
                rule_boxes.append(rule_box)

        rule_box_parent = Box(children=rule_boxes, layout=layout_rule_box)
        return rule_box_parent

    def create_rule_box(self, label_index: Optional[int]) -> Box:
        """Create box with the decision rules and the rule metrics

        :param label_index: index of the label in FeatureProcessor.labels
        :return: box with the rules and the rule metrics
        """
        rule_box_rules = self.create_rule_box_rules(label_index)
        rule_box_metrics = self.gen_rule_box_metrics(label_index)
        # layout_rule_box = Layout(
        #    margin="20px 0px 0px 0px", border="3px groove lightblue"
        # )
        layout_rule_box_all = Layout(
            margin="20px 20px 20px 20px", border="3px groove lightblue"
        )
        rule_box_all = HBox(
            children=[rule_box_rules, rule_box_metrics], layout=layout_rule_box_all
        )
        # rule_box_parent = Box(children=[rule_box_all], layout=layout_rule_box)
        return rule_box_all

    def create_rule_box_rules(self, label_index: int) -> VBox:
        """Create box with the decision rules

        :param label_index: index of the label in FeatureProcessor.labels
        :return:box with the decision rules
        """
        html_rule_caption = HTML(
            '<span style="font-weight:bold; font-size: 16px">'
            + "Rule for case duration >=\xa0"
            + str(self.current_threshold_numerical)
            + ":</span>",
            layout=Layout(margin="0px 0px 10px 0px"),
        )
        html_rules = self.create_pretty_html_rules(label_index)
        rules_html_widget = Box([HTML(value=html_rules)])

        def on_click_simplify_rules(b):
            try:
                button_simplify_rules.disabled = True
                button_elaborate_rules.disabled = True
                self.run_buttons[label_index].disabled = True
                self.dr_miners[label_index].simplify_rule_config()
                self.run_decision_miner(label_index)
                rule_box = self.create_rule_box(label_index)
                children = self.parent_rule_box.children
                children_first = children[:label_index]
                children_last = children[label_index + 1 :]

                children = list(children_first) + [rule_box] + list(children_last)
                self.parent_rule_box.children = children
            except MinimumValueReachedError:
                button_simplify_rules.disabled = True
            finally:
                self.run_buttons[label_index].disabled = False

        def on_click_elaborate_rules(b):
            try:
                button_simplify_rules.disabled = True
                button_elaborate_rules.disabled = True
                self.dr_miners[label_index].elaborate_rule_config()
                self.run_decision_miner(label_index)
                rule_box = self.create_rule_box(label_index)
                children = self.parent_rule_box.children
                children_first = children[:label_index]
                children_last = children[label_index + 1 :]

                children = list(children_first) + [rule_box] + list(children_last)
                self.parent_rule_box.children = children
            except MaximumValueReachedError:
                button_elaborate_rules.disabled = True
            finally:
                self.run_buttons[label_index].disabled = False

        button_simplify_rules = Button(description="Simplify rules")
        self.button_simplify_rules = button_simplify_rules
        button_simplify_rules.on_click(on_click_simplify_rules)
        if self.dr_miners[label_index].config_index == 0:
            button_simplify_rules.disabled = True
        button_elaborate_rules = Button(description="Elaborate rules")
        self.button_elaborate_rules = button_elaborate_rules
        button_elaborate_rules.on_click(on_click_elaborate_rules)
        if (
            self.dr_miners[label_index].config_index
            >= len(self.dr_miners[label_index].configs) - 1
        ):
            button_elaborate_rules.disabled = True

        hbox_change_rules = HBox(
            children=[button_simplify_rules, button_elaborate_rules]
        )
        vbox_rule = VBox(
            children=[html_rule_caption, rules_html_widget, hbox_change_rules]
        )

        return vbox_rule

    def run_decision_miner(self, label_index: int):
        """Run the decision rule miner to get decision rules

        :return:
        """
        self.dr_miners[label_index].run_pipeline()
        self.decision_rules[label_index] = self.dr_miners[label_index].structured_rules

    def create_pretty_html_rules(self, label_index) -> str:
        """Create html string with pretty decision rules

        :return: html string with pretty decision rules
        """
        pretty_rules = []
        for rule in self.decision_rules[label_index]:
            pretty_conds = []
            for cond in rule:
                attr = cond["attribute"]
                val = cond["value"]
                unequality = cond["unequal_sign"]
                if (
                    self.fp.attributes_dict[attr].attribute_data_type
                    == AttributeDataType.NUMERICAL
                ):
                    if unequality != "between":
                        pretty_str = attr + " " + unequality + "= " + val
                    else:
                        pretty_str = attr + " is in range " + val
                else:
                    if val == "1":
                        pretty_str = attr
                    else:
                        pretty_str = '<span style="color: Red">NOT</span> ' + attr
                pretty_conds.append(pretty_str)
            pretty_rule = ""
            for pretty_cond in pretty_conds:
                if pretty_rule != "":
                    pretty_rule = (
                        pretty_rule + '<span style="color: Green; font-weight: '
                        'bold;"><br>AND<br></span>'
                    )
                pretty_rule = pretty_rule + pretty_cond
            pretty_rule = (
                '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px;">'
                + pretty_rule
                + "</div>"
            )
            pretty_rules.append(pretty_rule)

        all_rules_html_text = ""
        for pretty_rule in pretty_rules:
            if all_rules_html_text != "":
                all_rules_html_text = (
                    all_rules_html_text
                    + '<div style="color: DodgerBlue; font-weight: bold; '
                    'margin-top: 5px; margin-bottom: 5px;">&emsp;OR</div>'
                )
            all_rules_html_text = all_rules_html_text + pretty_rule
        return all_rules_html_text

    def gen_rule_box_metrics(self, label_index: int) -> VBox:
        """Generate box that contains the metrics for the rules

        :param label_index: index of the label in FeatureProcessor.labels
        :return: box that contains the metrics for the rules
        """
        conf_matrix = self.gen_conf_matrix(label_index)
        avg_metrics = self.gen_avg_rule_metrics(label_index)
        metrics_box = VBox(
            [conf_matrix, avg_metrics], layout=Layout(margin="35px 0px 0px 30px")
        )
        return metrics_box

    def gen_conf_matrix(self, label_index: int):
        """Generate the box with the confusion matrix for the decision rules

        TODO: Exchange ipysheet with a box from ipywidgets as ipysheet is not working in
        Jupyter lab

        :return: box with the confusion matrix for the decision rules
        """
        header_color = "AliceBlue"
        cell_color = "Snow"
        font_size = "12px"
        html_rule_performance = HTML(
            '<span style="font-weight: bold; font-size:16px">Rule Performance</span>'
        )
        conf_matrix = ipysheet.sheet(
            rows=4, columns=4, column_headers=False, row_headers=False
        )
        ipysheet.cell(0, 0, "", read_only=True, background_color=header_color)
        ipysheet.cell(
            0,
            1,
            "Rule = True",
            read_only=True,
            style={"font-weight": "bold", "color": "Green", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            0,
            2,
            "Rule = False",
            read_only=True,
            style={"font-weight": "bold", "color": "Red", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            0,
            3,
            "Covered by rule",
            read_only=True,
            style={"font-weight": "bold", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            1,
            0,
            "High case duration",
            read_only=True,
            style={"font-weight": "bold", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            1,
            1,
            str(self.dr_miners[label_index].metrics["true_p"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            1,
            2,
            str(self.dr_miners[label_index].metrics["false_n"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            1,
            3,
            str(round(self.dr_miners[label_index].metrics["recall_p"] * 100)) + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            0,
            "Low case duration",
            read_only=True,
            style={"font-weight": "bold", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            2,
            1,
            str(self.dr_miners[label_index].metrics["false_p"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            2,
            str(self.dr_miners[label_index].metrics["true_n"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            3,
            str(round(self.dr_miners[label_index].metrics["recall_n"] * 100)) + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            3,
            0,
            "Rule correct",
            read_only=True,
            style={"font-weight": "bold", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            3,
            1,
            str(round(self.dr_miners[label_index].metrics["precision_p"] * 100)) + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            3,
            2,
            str(round(self.dr_miners[label_index].metrics["precision_n"] * 100)) + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(3, 3, "", read_only=True, background_color=cell_color)
        vbox_all = VBox(children=[html_rule_performance, conf_matrix])
        return vbox_all

    def gen_avg_rule_metrics(self, label_index: int) -> VBox:
        """Generate box with the average values of the dependent variable for cases
        for which the decision rules evaluate to true or false

        :param label_index: index of the label in FeatureProcessor.labels
        :return: box with the average rule metrics
        """
        label_unit = self.fp.labels[label_index].unit
        html_avg_true = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Rule = '
                    'True</span><br><span style="color: Red; font-size:16px">'
                    + str(round(self.dr_miners[label_index].metrics["avg_True"]))
                    + "\xa0"
                    + label_unit
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue", margin="0px 10px 0px 0px"
            ),
        )
        html_avg_false = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Rule = '
                    'False</span><br><span style="color: Green; font-size:16px">'
                    + str(round(self.dr_miners[label_index].metrics["avg_False"]))
                    + "\xa0"
                    + label_unit
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue",
                color="CornflowerBlue",
                margin="0px 0px 0px 10px",
            ),
        )
        hbox_metrics = HBox([html_avg_true, html_avg_false])
        html_avg_case_duration = HTML(
            '<span style="font-weight: bold; font-size:16px">Average case '
            "duration</span>"
        )
        vbox_metrics = VBox(
            [html_avg_case_duration, hbox_metrics],
            layout=Layout(margin="10px 0px 0px 0px"),
        )
        return vbox_metrics


class ValueSelection:
    """
    Creates the value selection box with which the user can select the values for
    which that are considered large.
    It is assumed that there is only one label (dependent variable).
    """

    def __init__(
        self,
        fp: FeatureProcessor,
        on_button_run_clicked: Callable,
        min_display_perc: float = 1.0,
        max_display_perc: float = 99.0,
        default_large_perc: int = 20,
    ):

        self.fp = fp
        self.on_button_run_clicked = on_button_run_clicked
        self.default_val = None
        self.min_val = None  # minimum value of dependent variable
        self.max_val = None  # maximum value of dependent variable
        # minimum percentage of data to display in cumulative probability plot
        self.min_display_perc = min_display_perc
        # maximum percentage of data to display in cumulative probability plot
        self.max_display_perc = max_display_perc
        # minimum value of dependent variable to display in cumulative probability plot
        self.min_display_val = None
        # maximum value of dependent variable to display in cumulative probability plot
        self.max_display_val = None
        self.default_long_perc = (
            default_large_perc  # default top percentage of long case duration
        )
        # the currently selected threshold
        self.current_th = None
        self.threshold_box = None
        self.button_run = None
        self.compute_statistics_from_df()
        self.selection_box = HBox()
        self.create_duration_selection_box()

    def get_selected_threshold(self) -> float:
        """Get the curently selected threshold

        :return: value of the currently selected threshold
        """
        return self.threshold_box.value

    def compute_statistics_from_df(self):
        """Set member variables:
        - minimum and maximum values of the dependent variable.
        - minimum and maximum display values of the dependent variable

        :return:
        """
        label_name = self.fp.labels[0].df_attribute_name
        self.default_val = self.fp.df[label_name].quantile(
            (100 - self.default_long_perc) / 100
        )
        self.min_val = self.fp.df[label_name].min()
        self.max_val = self.fp.df[label_name].max()
        self.min_display_val = self.fp.df[label_name].quantile(
            self.min_display_perc / 100
        )
        self.max_display_val = self.fp.df[label_name].quantile(
            self.max_display_perc / 100
        )

    def create_duration_selection_box(self):
        """Create the box for the duration selection

        :return:
        """
        label_title = Label("Define high case duration:")

        label_description = Label("Case duration >=\xa0")

        label_unit = Label("\xa0" + self.fp.labels[0].unit, layout=Layout(width="auto"))
        label_percentage = Label(
            "\xa0 (Top\xa0"
            + str(
                self.get_percentile(
                    self.fp.df[self.fp.labels[0].df_attribute_name], self.default_val
                )
            )
            + "%)",
            layout=Layout(width="auto"),
        )
        selection_box_layout = Layout(max_width="80px")
        selection_box_number = widgets.BoundedIntText(
            value=self.default_val,
            min=self.min_val,
            max=self.max_val,
            step=1,
            layout=selection_box_layout,
        )
        self.threshold_box = selection_box_number

        def handle_label_percentage_description(change):
            new_description = (
                "\xa0 (Top\xa0"
                + str(
                    self.get_percentile(
                        self.fp.df[self.fp.labels[0].df_attribute_name], change.new
                    )
                )
                + "%)"
            )
            label_percentage.value = new_description

        selection_box_number.observe(handle_label_percentage_description, names="value")

        # Default value Button
        def on_button_default_clicked(b):
            selection_box_number.value = self.default_val

        button_default = Button(
            description="Default: Top\xa0" + str(self.default_long_perc) + "%"
        )
        button_default.on_click(on_button_default_clicked)

        hbox_selection_right = HBox(
            children=[selection_box_number, label_unit, label_percentage]
        )
        vbox_selection_right = VBox(children=[hbox_selection_right, button_default])
        hbox_selection = HBox(children=[label_description, vbox_selection_right])

        button_run = Button(description="Mine rules!")
        self.button_run = button_run
        button_clicked_partial = functools.partial(
            self.on_button_run_clicked, label_index=0
        )
        button_run.on_click(button_clicked_partial)

        vbox_run_button_layout = Layout(
            flex="1", justify_content="flex-end", align_items="center"
        )
        vbox_run_button = VBox([button_run], layout=vbox_run_button_layout)

        vbox_duration_selection_layout = Layout(min_width="350px")
        vbox_duration_selection = VBox(
            children=[label_title, hbox_selection, vbox_run_button],
            layout=vbox_duration_selection_layout,
        )
        prob_figure_widget = self.create_probability_figure_widget()
        self.selection_box.children = [vbox_duration_selection, prob_figure_widget]

    def get_percentile(self, series: pd.Series, val: float):
        """Get the percentile of a value in a series

        :param series: series with numerical values
        :param val: value for which to compute the percentile
        :return: the percentile of the value in the series
        """
        return round(100 - stats.percentileofscore(series, val))

    def create_probability_figure_widget(self) -> go.FigureWidget:
        """Create the figure for the cumulative probability of the dependent
        variable

        :return: FigureWidget object with the figure
        """
        label_name = self.fp.labels[0].df_attribute_name
        df_float = pd.DataFrame(self.fp.df[label_name].astype(float))
        fig = px.ecdf(df_float, x=label_name)
        fig.update_layout(
            {
                "xaxis_title": "Case duration (days)",
                "yaxis_title": "cumulative probability",
            }
        )
        fig.update_xaxes(range=[self.min_display_val, self.max_display_val])
        fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        fig_widget = go.FigureWidget(fig)
        return fig_widget
