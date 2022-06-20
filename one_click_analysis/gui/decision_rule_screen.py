import functools
from typing import Callable
from typing import List

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
from one_click_analysis.feature_processing.attributes.attribute import Attribute
from one_click_analysis.feature_processing.attributes.attribute import AttributeDataType
from one_click_analysis.feature_processing.attributes.feature import Feature
from one_click_analysis.gui.feature_selection import FeatureSelection


class DecisionRulesScreen:
    def __init__(
        self,
        df: pd.DataFrame,
        attributes: List[Attribute],
        features: List[Feature],
        target_features: List[Feature],
    ):
        """
        :param df: DataFrame with feature and target columns
        :param fp: FeatureProcessor with processed features
        """
        self.df = df
        self.attributes = attributes
        # Concatenate df_x and df_target
        self.df_concat = pd.concat
        self.features = features
        self.target_features = target_features
        self.feature_names = self._create_feature_names()
        # map feature names to values (0 to n)
        self.target_feature_map = self._create_target_feature_map()
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
        self.feature_selection_box = self._create_feature_selection_box()
        self.value_selection = self._create_ValueSelection()
        self.parent_rule_box = self._init_rules_parent_box()

    def _create_feature_selection_box(self):
        """Create feature selection box"""
        feature_selection = FeatureSelection(
            features=self.features, attributes=self.attributes
        )
        return feature_selection.selection_box

    def _create_feature_names(self):
        """Generate list of feature names"""
        attribute_names = []
        for f in self.features:
            attribute_names.append(f.df_column_name)
        return attribute_names

    def _create_target_feature_map(self):
        target_feature_map = {}
        for i, tf in enumerate(self.target_features):
            target_feature_map[tf.df_column_name] = i

        return target_feature_map

    def _set_numerical(self):
        """Set the is_numerical member variable

        :return: True if label is numerical and False if labels are all categorical
        """
        datatypes = [tf.datatype for tf in self.target_features]
        if len(datatypes) == 1:
            if datatypes[0] == AttributeDataType.NUMERICAL:
                return True
            else:
                return False
        elif (
            len(self.target_features) > 1
            and AttributeDataType.NUMERICAL not in datatypes
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
            value_selection = ValueSelection(
                df=self.df,
                target_feature=self.target_features[0],
                on_button_run_clicked=self.on_button_run_clicked,
                min_display_perc=1.0,
                max_display_perc=99.0,
                default_large_perc=20,
            )
            self.run_buttons[
                self.target_features[0].df_column_name
            ] = value_selection.button_run
            return value_selection
        else:
            return None

    def update_features(self, features: List[Feature]):
        """Define behaviour when the attribute selection is updated. Here, the screen is
        simply constructed again with the new attributes.

        :param features: List with features
        :return:
        """
        self.features = features
        self.feature_names = self._create_feature_names()
        self.parent_rule_box = self._init_rules_parent_box()
        self.dr_miners = {}
        self.current_threshold_numerical = None

        self.create_decision_rule_screen()

    def create_decision_rule_screen(self):
        """Create and get the decision rule screen, i.e. the box that contains the
        selection box and the rule box

        :return: box with the decision rule screen
        """
        if self.is_numerical:
            self.decision_rule_box.children = [
                self.feature_selection_box,
                self.value_selection.selection_box,
                self.parent_rule_box,
            ]
        else:
            self.decision_rule_box.children = [
                self.feature_selection_box,
                self.parent_rule_box,
            ]

    def on_button_run_clicked(self, b, target_feature: Feature):
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
        target_col_name = target_feature.df_column_name
        self.run_buttons[target_col_name].disabled = True

        self.dr_miners[target_col_name] = DecisionRuleMiner(
            self.df,
            target_col_name,
            self.feature_names,
            pos_class=pos_class,
            threshold=threshold,
            k=99,
        )
        self.run_decision_miner(target_feature)
        rule_box = self.create_rule_box(target_feature)
        children = self.parent_rule_box.children
        target_index = self.target_feature_map[target_col_name]
        children_first = children[:target_index]
        children_last = children[target_index + 1 :]

        children = list(children_first) + [rule_box] + list(children_last)
        self.parent_rule_box.children = children
        self.run_buttons[target_col_name].disabled = False

    def get_percentile(self, series: pd.Series, val: float):
        """Get the percentile of a value in a series

        :param series: series with numerical values
        :param val: value for which to compute the percentile
        :return: the percentile of the value in the series
        """
        return round(100 - stats.percentileofscore(series, val))

    def _gen_rule_caption(self, target_str: str):
        layout = Layout(margin="0px 0px 10px 0px")
        if self.is_numerical:
            html_rule_caption = HTML(
                '<span style="font-weight:bold; font-size: 16px">'
                + "Rule for "
                + target_str
                + ">=\xa0"
                + str(self.current_threshold_numerical)
                + ":</span>",
                layout=layout,
            )
        else:
            html_rule_caption = HTML(
                '<span style="font-weight:bold; font-size: 16px">'
                + "Rule for "
                + target_str
                + "</span>",
                layout=layout,
            )
        return html_rule_caption

    def _init_rules_parent_box(self):
        layout_rule_box_parent = Layout(
            margin="20px 0px 0px 0px", border="3px groove lightblue"
        )

        rule_boxes = []
        if self.is_numerical:
            rule_box = HBox(
                children=[
                    self._gen_rule_caption(self.target_features[0].df_column_name)
                ]
            )
            rule_boxes.append(rule_box)
        else:
            for tf in self.target_features:
                tf_col_name = tf.df_column_name
                button_run = Button(description="Mine rules!")

                button_clicked_partial = functools.partial(
                    self.on_button_run_clicked, target_feature=tf
                )
                button_run.on_click(button_clicked_partial)

                self.run_buttons[tf_col_name] = button_run
                # vbox_run_button_layout = Layout(
                #    flex="1", justify_content="flex-end", align_items="center"
                # )
                vbox_run_button = VBox([button_run])  # , layout=vbox_run_button_layout)
                layout_rule_box = Layout(
                    margin="20px 20px 20px 20px", border="3px groove lightblue"
                )
                rule_box = VBox(
                    children=[self._gen_rule_caption(tf_col_name), vbox_run_button],
                    layout=layout_rule_box,
                )
                rule_boxes.append(rule_box)

        rule_box_parent = VBox(children=rule_boxes, layout=layout_rule_box_parent)
        return rule_box_parent

    def create_rule_box(self, target_feature: Feature) -> Box:
        """Create box with the decision rules and the rule metrics

        :param label_index: index of the label in FeatureProcessor.labels
        :return: box with the rules and the rule metrics
        """
        rule_box_rules = self.create_rule_box_rules(target_feature)
        rule_box_metrics = self.gen_rule_box_metrics(target_feature)
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

    def create_rule_box_rules(self, target_feature: Feature) -> VBox:
        """Create box with the decision rules

        :param label_index: index of the label in FeatureProcessor.labels
        :return:box with the decision rules
        """
        target_feature_col_name = target_feature.df_column_name
        label_str = target_feature_col_name
        html_rule_caption = self._gen_rule_caption(target_str=label_str)
        html_rules = self.create_pretty_html_rules(target_feature)
        rules_html_widget = Box([HTML(value=html_rules)])
        target_index = self.target_feature_map[target_feature_col_name]

        def on_click_simplify_rules(b):
            try:
                button_simplify_rules.disabled = True
                button_elaborate_rules.disabled = True
                self.run_buttons[target_feature_col_name].disabled = True
                self.dr_miners[target_feature_col_name].simplify_rule_config()
                self.run_decision_miner(target_feature)
                rule_box = self.create_rule_box(target_feature)
                children = self.parent_rule_box.children
                children_first = children[:target_index]
                children_last = children[target_index + 1 :]

                children = list(children_first) + [rule_box] + list(children_last)
                self.parent_rule_box.children = children
            except MinimumValueReachedError:
                button_simplify_rules.disabled = True
            finally:
                self.run_buttons[target_feature_col_name].disabled = False

        def on_click_elaborate_rules(b):
            try:
                button_simplify_rules.disabled = True
                button_elaborate_rules.disabled = True
                self.dr_miners[target_feature_col_name].elaborate_rule_config()
                self.run_decision_miner(target_feature)
                rule_box = self.create_rule_box(target_feature)
                children = self.parent_rule_box.children
                children_first = children[:target_index]
                children_last = children[target_index + 1 :]

                children = list(children_first) + [rule_box] + list(children_last)
                self.parent_rule_box.children = children
            except MaximumValueReachedError:
                button_elaborate_rules.disabled = True
            finally:
                self.run_buttons[target_feature_col_name].disabled = False

        button_simplify_rules = Button(description="Simplify rules")
        self.buttons_simplify_rules[target_feature_col_name] = button_simplify_rules
        button_simplify_rules.on_click(on_click_simplify_rules)
        if self.dr_miners[target_feature_col_name].config_index == 0:
            button_simplify_rules.disabled = True
        button_elaborate_rules = Button(description="Elaborate rules")
        self.buttons_elaborate_rules[target_feature_col_name] = button_elaborate_rules
        button_elaborate_rules.on_click(on_click_elaborate_rules)
        if (
            self.dr_miners[target_feature_col_name].config_index
            >= len(self.dr_miners[target_feature_col_name].configs) - 1
        ):
            button_elaborate_rules.disabled = True

        hbox_change_rules = HBox(
            children=[button_simplify_rules, button_elaborate_rules]
        )
        vbox_rule = VBox(
            children=[html_rule_caption, rules_html_widget, hbox_change_rules]
        )

        return vbox_rule

    def run_decision_miner(self, target_feature: Feature):
        """Run the decision rule miner to get decision rules

        :return:
        """
        target_feature_col_name = target_feature.df_column_name
        self.dr_miners[target_feature_col_name].run_pipeline()
        self.decision_rules[target_feature_col_name] = self.dr_miners[
            target_feature_col_name
        ].structured_rules

    def create_pretty_html_rules(self, target_feature: Feature) -> str:
        """Create html string with pretty decision rules

        :return: html string with pretty decision rules
        """
        feature_dict = {f.df_column_name: f for f in self.features}
        pretty_rules = []
        for rule in self.decision_rules[target_feature.df_column_name]:
            pretty_conds = []
            for cond in rule:
                feature_name = cond["attribute"]
                val = cond["value"]
                unequality = cond["unequal_sign"]
                if feature_dict[feature_name].datatype == AttributeDataType.NUMERICAL:
                    if unequality != "between":
                        pretty_str = feature_name + " " + unequality + "= " + val
                    else:
                        pretty_str = feature_name + " is in range " + val
                else:
                    if val == "1":
                        pretty_str = feature_name
                    else:
                        pretty_str = (
                            '<span style="color: Red">NOT</span> ' + feature_name
                        )
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

    def gen_rule_box_metrics(self, target_feature: Feature) -> VBox:
        """Generate box that contains the metrics for the rules

        :param label_index: index of the label in FeatureProcessor.labels
        :return: box that contains the metrics for the rules
        """
        conf_matrix = self.gen_conf_matrix(target_feature)
        if self.is_numerical:
            avg_metrics = self.gen_avg_rule_metrics(target_feature)
            metrics_box = VBox(
                [conf_matrix, avg_metrics], layout=Layout(margin="35px 0px 0px 30px")
            )
        else:
            metrics_box = VBox([conf_matrix], layout=Layout(margin="35px 0px 0px 30px"))

        return metrics_box

    def gen_conf_matrix(self, target_feature: Feature):
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
        target_col_name = target_feature.df_column_name
        if self.is_numerical:
            pos_label_str = "High " + target_col_name
            neg_label_str = "Low " + target_col_name
        else:
            pos_label_str = target_col_name
            neg_label_str = "Not " + target_col_name

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
            pos_label_str,
            read_only=True,
            style={"font-weight": "bold", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            1,
            1,
            str(self.dr_miners[target_col_name].metrics["true_p"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            1,
            2,
            str(self.dr_miners[target_col_name].metrics["false_n"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            1,
            3,
            str(round(self.dr_miners[target_col_name].metrics["recall_p"] * 100)) + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            0,
            neg_label_str,
            read_only=True,
            style={"font-weight": "bold", "font-size": font_size},
            background_color=header_color,
        )
        ipysheet.cell(
            2,
            1,
            str(self.dr_miners[target_col_name].metrics["false_p"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            2,
            str(self.dr_miners[target_col_name].metrics["true_n"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            3,
            str(round(self.dr_miners[target_col_name].metrics["recall_n"] * 100)) + "%",
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
            str(round(self.dr_miners[target_col_name].metrics["precision_p"] * 100))
            + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            3,
            2,
            str(round(self.dr_miners[target_col_name].metrics["precision_n"] * 100))
            + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(3, 3, "", read_only=True, background_color=cell_color)
        vbox_all = VBox(children=[html_rule_performance, conf_matrix])
        return vbox_all

    def gen_avg_rule_metrics(self, target_feature: Feature) -> VBox:
        """Generate box with the average values of the dependent variable for cases
        for which the decision rules evaluate to true or false

        :param label_index: index of the label in FeatureProcessor.labels
        :return: box with the average rule metrics
        """
        target_unit = target_feature.unit
        target_col_name = target_feature.df_column_name
        html_avg_true = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Rule = '
                    'True</span><br><span style="color: Red; font-size:16px">'
                    + str(round(self.dr_miners[target_col_name].metrics["avg_True"]))
                    + "\xa0"
                    + target_unit
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
                    + str(round(self.dr_miners[target_col_name].metrics["avg_False"]))
                    + "\xa0"
                    + target_unit
                    + "</span></center>"
                )
            ],
            layout=Layout(
                border="3px double CornflowerBlue",
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
    It is assumed that there is only one target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_feature: Feature,
        on_button_run_clicked: Callable,
        min_display_perc: float = 1.0,
        max_display_perc: float = 99.0,
        default_large_perc: int = 20,
    ):
        self.df = df
        self.target_feature = target_feature
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
            default_large_perc  # default top percentage of large value
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
        target_name = self.target_feature.df_column_name
        self.default_val = self.df[target_name].quantile(
            (100 - self.default_long_perc) / 100
        )
        self.min_val = self.df[target_name].min()
        self.max_val = self.df[target_name].max()
        self.min_display_val = self.df[target_name].quantile(
            self.min_display_perc / 100
        )
        self.max_display_val = self.df[target_name].quantile(
            self.max_display_perc / 100
        )

    def create_duration_selection_box(self):
        """Create the box for the duration selection

        :return:
        """
        target_name = self.target_feature.df_column_name
        target_unit = self.target_feature.unit
        label_title = Label("Define high " + target_name + ":")

        label_description = Label(target_name + " >=\xa0")

        label_unit = Label("\xa0" + target_unit, layout=Layout(width="auto"))
        label_percentage = Label(
            "\xa0 (Top\xa0"
            + str(self.get_percentile(self.df[target_name], self.default_val))
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
            target_col_name = self.target_feature.df_column_name
            new_description = (
                "\xa0 (Top\xa0"
                + str(self.get_percentile(self.df[target_col_name], change.new))
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
            self.on_button_run_clicked, target_feature=self.target_feature
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
        target_col_name = self.target_feature.df_column_name
        target_unit = self.target_feature.unit
        df_float = pd.DataFrame(self.df[target_col_name].astype(float))
        fig = px.ecdf(df_float, x=target_col_name)
        unit = target_unit
        if unit != "":
            unit_str = " (" + unit.lower() + ")"
        else:
            unit_str = ""
        xaxis_title = target_col_name + unit_str
        fig.update_layout(
            {
                "xaxis_title": xaxis_title,
                "yaxis_title": "cumulative probability",
            }
        )
        fig.update_xaxes(range=[self.min_display_val, self.max_display_val])
        fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        fig_widget = go.FigureWidget(fig)
        return fig_widget
