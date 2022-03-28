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

from decision_rules.decision_rule_miner import DecisionRuleMiner
from errors import MaximumValueReachedError
from errors import MinimumValueReachedError
from feature_processing.attributes import AttributeDataType
from feature_processing.feature_processor import FeatureProcessor


class DecisionRulesScreen:
    def __init__(self, fp: FeatureProcessor, pos_class=None):
        """

        :param fp: FeatureProcessor with processed features
        :param pos_class: the value of the positive class of the dependent variable.
        Leave at None if class is numerical
        """
        self.fp = fp
        self.df = fp.df
        self.label = fp.label.df_attribute_name
        self.label_unit = fp.label.unit
        # the value of the positive class. Leave at None if class is numerical
        self.pos_class = pos_class
        self.default_long_perc = 20  # default top percentage of long case duration
        # minimum percentage of data to display in cumulative probability plot
        self.min_display_perc = 1
        # maximum percentage of data to display in cumulative probability plot
        self.max_display_perc = 99
        self.default_val = None
        self.min_val = None  # minimum value of dependent variable
        self.max_val = None  # maximum value of dependent variable
        # minimum value of dependent variable to display in cumulative probability plot
        self.min_display_val = None
        # maximum value of dependent variable to display in cumulative probability plot
        self.max_display_val = None
        self.get_statistics_from_df()
        self.high_duration_box = None
        self.dr_miner = None
        self.decision_rules = None
        self.current_case_duration = None
        self.rule_box = HBox([Label("Hello")])
        self.decision_rule_screen = None
        self.button_run = None
        self.button_elaborate_rules = None
        self.button_simplify_rules = None

    def get_statistics_from_df(self):
        """Set member variables:
        - minimum and maximum values of the dependent variable.
        - minimum and maximum display values of the dependent variable

        :return:
        """
        self.default_val = self.df[self.label].quantile(
            (100 - self.default_long_perc) / 100
        )
        self.min_val = self.df[self.label].min()
        self.max_val = self.df[self.label].max()
        self.min_display_val = self.df[self.label].quantile(self.min_display_perc / 100)
        self.max_display_val = self.df[self.label].quantile(self.max_display_perc / 100)

    def create_decision_rule_screen(self) -> VBox:
        """create and get the decision rule screen, i.e. the box that contains the
        selection box and the rule box

        :return: box with the decision rule screen
        """
        selection_box = self.create_duration_selection_box()
        decision_rule_box = VBox(children=[selection_box, self.rule_box])
        self.decision_rule_screen = decision_rule_box
        return decision_rule_box

    def create_duration_selection_box(self) -> HBox:
        """Create the box for the duration selection

        :return: the box for the duration selection
        """
        label_title = Label("Define high case duration:")

        label_description = Label("Case duration >=\xa0")

        label_unit = Label("\xa0" + self.label_unit, layout=Layout(width="auto"))
        label_percentage = Label(
            "\xa0 (Top\xa0"
            + str(self.get_percentile(self.df[self.label], self.default_val))
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
        self.high_duration_box = selection_box_number

        def handle_label_percentage_description(change):
            new_description = (
                "\xa0 (Top\xa0"
                + str(self.get_percentile(self.df[self.label], change.new))
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

        # Run button
        def on_button_run_clicked(b):
            # secure the current content of the box

            case_duration_th = self.high_duration_box.value
            # Do not run the miner twice for the same threshold value
            if case_duration_th == self.current_case_duration:
                return
            button_run.disabled = True
            if (
                self.dr_miner is None
                or self.dr_miner.threshold != self.high_duration_box.value
            ):
                self.dr_miner = DecisionRuleMiner(
                    self.df,
                    self.label,
                    self.fp.attributes_dict.keys(),
                    pos_class=None,
                    threshold=self.high_duration_box.value,
                )
            self.run_decision_miner()
            self.current_case_duration = case_duration_th
            self.rule_box = self.create_rule_box()
            self.decision_rule_screen.children = [
                self.decision_rule_screen.children[0]
            ] + [self.rule_box]
            button_run.disabled = False

        button_run = Button(description="Mine rules!")
        self.button_run = button_run
        button_run.on_click(on_button_run_clicked)

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
        hbox_all = HBox(children=[vbox_duration_selection, prob_figure_widget])
        return hbox_all

    def get_percentile(self, series: pd.Series, val: float):
        """Get the percentile of a value in a series

        :param series: series with numerical values
        :param val: value for which to compute the percentile
        :return: the percentile of the value in the series
        """
        return round(100 - stats.percentileofscore(series, val))

    def create_probability_figure_widget(self) -> go.FigureWidget:
        """Create the figure for the cumulative probability of the dependent variable

        :return: FigureWidget object with the figure
        """
        df_float = pd.DataFrame(self.df[self.label].astype(float))
        fig = px.ecdf(df_float, x=self.label)
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

    def create_rule_box(self) -> Box:
        """Create box with the decision rules and the rule metrics

        :return: box with the rules and the rule metrics
        """
        rule_box_rules = self.create_rule_box_rules()
        rule_box_metrics = self.gen_rule_box_metrics()
        layout_rule_box = Layout(
            margin="20px 0px 0px 0px", border="3px groove lightblue"
        )
        layout_rule_box_all = Layout(
            margin="20px 20px 20px 20px", border="3px groove lightblue"
        )
        rule_box_all = HBox(
            children=[rule_box_rules, rule_box_metrics], layout=layout_rule_box_all
        )
        rule_box_parent = Box(children=[rule_box_all], layout=layout_rule_box)
        return rule_box_parent

    def create_rule_box_rules(self) -> VBox:
        """Create box with the decision rules

        :return:box with the decision rules
        """
        html_rule_caption = HTML(
            '<span style="font-weight:bold; font-size: 16px">'
            + "Rule for case duration >=\xa0"
            + str(self.current_case_duration)
            + ":</span>",
            layout=Layout(margin="0px 0px 10px 0px"),
        )
        html_rules = self.create_pretty_html_rules()
        rules_html_widget = Box([HTML(value=html_rules)])

        def on_click_simplify_rules(b):
            try:
                button_simplify_rules.disabled = True
                button_elaborate_rules.disabled = True
                self.button_run.disabled = True
                self.dr_miner.simplify_rule_config()
                self.run_decision_miner()
                self.rule_box = self.create_rule_box()
                self.decision_rule_screen.children = [
                    self.decision_rule_screen.children[0]
                ] + [self.rule_box]
            except MinimumValueReachedError:
                button_simplify_rules.disabled = True
            finally:
                self.button_run.disabled = False

        def on_click_elaborate_rules(b):
            try:
                button_simplify_rules.disabled = True
                button_elaborate_rules.disabled = True
                self.dr_miner.elaborate_rule_config()
                self.run_decision_miner()
                self.rule_box = self.create_rule_box()
                self.decision_rule_screen.children = [
                    self.decision_rule_screen.children[0]
                ] + [self.rule_box]
            except MaximumValueReachedError:
                button_elaborate_rules.disabled = True
            finally:
                self.button_run.disabled = False

        button_simplify_rules = Button(description="Simplify rules")
        self.button_simplify_rules = button_simplify_rules
        button_simplify_rules.on_click(on_click_simplify_rules)
        if self.dr_miner.config_index == 0:
            button_simplify_rules.disabled = True
        button_elaborate_rules = Button(description="Elaborate rules")
        self.button_elaborate_rules = button_elaborate_rules
        button_elaborate_rules.on_click(on_click_elaborate_rules)
        if self.dr_miner.config_index >= len(self.dr_miner.configs) - 1:
            button_elaborate_rules.disabled = True

        hbox_change_rules = HBox(
            children=[button_simplify_rules, button_elaborate_rules]
        )
        vbox_rule = VBox(
            children=[html_rule_caption, rules_html_widget, hbox_change_rules]
        )

        return vbox_rule

    def run_decision_miner(self):
        """Run the decision rule miner to get decision rules

        :return:
        """
        self.dr_miner.run_pipeline()
        self.decision_rules = self.dr_miner.structured_rules

    def create_pretty_html_rules(self) -> str:
        """Create html string with pretty decision rules

        :return: html string with pretty decision rules
        """
        pretty_rules = []
        for rule in self.decision_rules:
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

    def gen_rule_box_metrics(self) -> VBox:
        """Generate box that contains the metrics for the rules

        :return: box that contains the metrics for the rules
        """
        conf_matrix = self.gen_conf_matrix()
        avg_metrics = self.gen_avg_rule_metrics()
        metrics_box = VBox(
            [conf_matrix, avg_metrics], layout=Layout(margin="35px 0px 0px 30px")
        )
        return metrics_box

    def gen_conf_matrix(self):
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
            str(self.dr_miner.metrics["true_p"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            1,
            2,
            str(self.dr_miner.metrics["false_n"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            1,
            3,
            str(round(self.dr_miner.metrics["recall_p"] * 100)) + "%",
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
            str(self.dr_miner.metrics["false_p"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            2,
            str(self.dr_miner.metrics["true_n"]),
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            2,
            3,
            str(round(self.dr_miner.metrics["recall_n"] * 100)) + "%",
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
            str(round(self.dr_miner.metrics["precision_p"] * 100)) + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(
            3,
            2,
            str(round(self.dr_miner.metrics["precision_n"] * 100)) + "%",
            read_only=True,
            style={"font-size": font_size},
            background_color=cell_color,
        )
        ipysheet.cell(3, 3, "", read_only=True, background_color=cell_color)
        vbox_all = VBox(children=[html_rule_performance, conf_matrix])
        return vbox_all

    def gen_avg_rule_metrics(self) -> VBox:
        """Generate box with the average values of the dependent variable for cases
        for which the decision rules evaluate to true or false

        :return: box with the average rule metrics
        """
        html_avg_true = Box(
            [
                HTML(
                    '<center><span style="font-weight:bold"> Rule = '
                    'True</span><br><span style="color: Red; font-size:16px">'
                    + str(round(self.dr_miner.metrics["avg_True"]))
                    + "\xa0"
                    + self.label_unit
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
                    + str(round(self.dr_miner.metrics["avg_False"]))
                    + "\xa0"
                    + self.label_unit
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
