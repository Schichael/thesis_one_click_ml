import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout, Label, Text, GridBox, GridspecLayout, Dropdown, Tab, Button, \
    BoundedIntText, \
    HTML, Box
import queries
import plotly.express as px
from pycelonis import get_celonis
from DataModel import DataModelInfo
import plotly.graph_objects as go
import math
from preprocessing import Attribute, AttributeDataType, Preprocessor
from pycelonis.celonis_api.pql.pql import PQL, PQLColumn, PQLFilter
import functools
import logging
import sys
import numpy as np

from sklearn.metrics import confusion_matrix
from pandas.api.types import is_numeric_dtype
import pandas as pd
from DecisionRuleMiner import DecisionRuleMiner
from scipy import stats
from errors import MinimumValueReachedError, MaximumValueReachedError
import ipysheet


class DecisionRulesBox:
    def __init__(self, df, label, label_unit, attributes_dict, pos_class=None):
        self.df = df
        self.label = label
        self.label_unit = label_unit
        self.attributes_dict = attributes_dict
        self.pos_class = pos_class  # the value of the positive class. Leave at None if class is numerical
        self.default_long_perc = 20  # default top percentage of long case duration
        self.min_display_perc = 1  # minimum percentage of data to display in plot
        self.max_display_perc = 99  # maximum percentage of data to display in plot
        self.default_val = None
        self.min_val = None
        self.max_val = None
        self.min_display_val = None
        self.max_display_val = None
        self.get_statistics_from_df()
        self.high_duration_box = None
        self.dr_miner = None
        self.decision_rules = None
        self.current_case_duration = None
        self.rule_box = HBox([Label("Hello")])
        self.view = None
        self.button_run = None
        self.button_elaborate_rules = None
        self.button_simplify_rules = None

    def get_statistics_from_df(self):
        self.default_val = self.df[self.label].quantile((100 - self.default_long_perc) / 100)
        self.min_val = self.df[self.label].min()
        self.max_val = self.df[self.label].max()
        self.min_display_val = self.df[self.label].quantile(self.min_display_perc / 100)
        self.max_display_val = self.df[self.label].quantile(self.max_display_perc / 100)

    def create_view(self):
        selection_box = self.create_duration_selection_box()
        vbox_view = VBox(children=[selection_box, self.rule_box])
        self.view = vbox_view
        return vbox_view

    def create_duration_selection_box(self):
        label_title = Label("Define high case duration:")

        label_description = Label("Case duration >=\xa0")

        label_unit = Label("\xa0" + self.label_unit, layout=Layout(width='auto'))
        label_percentage = Label(
            "\xa0 (Top\xa0" + str(self.get_percentile_score(self.df[self.label], self.default_val)) + "%)",
            layout=Layout(width='auto'))
        selection_box_layout = Layout(max_width="80px")
        selection_box_number = widgets.BoundedIntText(value=self.default_val, min=self.min_val, max=self.max_val,
                                                      step=1, layout=selection_box_layout)
        self.high_duration_box = selection_box_number

        def handle_label_percentage_description(change):
            new_description = "\xa0 (Top\xa0" + str(self.get_percentile_score(self.df[self.label], change.new)) + "%)"
            label_percentage.value = new_description

        selection_box_number.observe(handle_label_percentage_description, names="value")

        # Default value Button
        def on_button_default_clicked(b):
            selection_box_number.value = self.default_val

        button_default = Button(description="Default: Top\xa0" + str(self.default_long_perc) + "%")
        button_default.on_click(on_button_default_clicked)

        hbox_selection_right = HBox(children=[selection_box_number, label_unit, label_percentage])
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
            if self.dr_miner is None or self.dr_miner.threshold != self.high_duration_box.value:
                self.dr_miner = DecisionRuleMiner(self.df, self.label, self.attributes_dict.keys(), pos_class=None,
                                                  threshold=self.high_duration_box.value)
            self.run_decision_miner()
            self.current_case_duration = case_duration_th
            self.rule_box = self.create_rule_box()
            self.view.children = [self.view.children[0]] + [self.rule_box]
            button_run.disabled = False

        button_run = Button(description="Mine rules!")
        self.button_run = button_run
        button_run.on_click(on_button_run_clicked)

        vbox_run_button_layout = Layout(flex='1', justify_content='flex-end', align_items='center')
        vbox_run_button = VBox([button_run], layout=vbox_run_button_layout)

        vbox_duration_selection_layout = Layout(min_width="350px")
        vbox_duration_selection = VBox(children=[label_title, hbox_selection, vbox_run_button],
                                       layout=vbox_duration_selection_layout)
        prob_figure_widget = self.create_probability_figure_widget()
        hbox_all = HBox(children=[vbox_duration_selection, prob_figure_widget])
        return hbox_all

    def get_percentile_score(self, series, val):
        return round(100 - stats.percentileofscore(series, val))

    def create_probability_figure_widget(self):
        df_float = pd.DataFrame(self.df[self.label].astype(float))
        fig = px.ecdf(df_float, x=self.label)
        fig.update_layout({'xaxis_title': "Case duration (days)", 'yaxis_title': "cumulative probability"})
        fig.update_xaxes(range=[self.min_display_val, self.max_display_val])
        fig.update_layout(height=300, margin={'l': 10, 'r': 10, 't': 40, 'b': 10})
        fig_widget = go.FigureWidget(fig)
        return fig_widget

    def create_rule_box(self):
        rule_box_rules = self.create_rule_box_rules()
        rule_box_metrics = self.gen_rule_box_metrics()
        layout_rule_box = Layout(margin='20px 0px 0px 0px', border='3px groove lightblue')
        layout_rule_box_all = Layout(margin='20px 20px 20px 20px', border='3px groove lightblue')
        rule_box_all = HBox(children=[rule_box_rules, rule_box_metrics], layout=layout_rule_box_all)
        rule_box_parent = Box(children=[rule_box_all], layout=layout_rule_box)
        return rule_box_parent

    def create_rule_box_rules(self):
        html_rule_caption = HTML(
            "<span style=\"font-weight:bold; font-size: 16px\">" + "Rule for case duration >=\xa0" + str(
                self.current_case_duration) + ":</span>", layout=Layout(margin='0px 0px 10px 0px'))
        html_rules = self.get_pretty_html_rules()
        rules_html_widget = Box([HTML(value=html_rules)])

        def on_click_simplify_rules(b):
            try:
                button_simplify_rules.disabled = True
                button_elaborate_rules.disabled = True
                self.button_run.disabled = True
                self.dr_miner.simplify_rule_config()
                self.run_decision_miner()
                self.rule_box = self.create_rule_box()
                self.view.children = [self.view.children[0]] + [self.rule_box]
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
                self.view.children = [self.view.children[0]] + [self.rule_box]
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

        hbox_change_rules = HBox(children=[button_simplify_rules, button_elaborate_rules])
        vbox_rule = VBox(children=[html_rule_caption, rules_html_widget, hbox_change_rules])

        return vbox_rule

    def run_decision_miner(self):
        self.dr_miner.run_pipeline()
        self.decision_rules = self.dr_miner.structured_rules

    def enable_buttons(self):
        self.button_elaborate_rules.disabled = False
        self.button_simplify_rules.disabled = False
        self.button_run.disabled = False

    def disable_buttons(self):
        self.button_elaborate_rules.disabled = True
        self.button_simplify_rules.disabled = True
        self.button_run.disabled = True

    def get_pretty_html_rules(self):
        pretty_rules = []
        for rule in self.decision_rules:
            pretty_conds = []
            for cond in rule:
                attr = cond['attribute']
                val = cond['value']
                unequality = cond['unequal_sign']
                if self.attributes_dict[attr].attribute_data_type == AttributeDataType.NUMERICAL:
                    if unequality != "between":
                        pretty_str = attr + " " + unequality + "= " + val
                    else:
                        pretty_str = attr + " is in range " + val
                else:
                    if val == "1":
                        pretty_str = attr
                    else:
                        pretty_str = "<span style=\"color: Red\">NOT</span> " + attr
                pretty_conds.append(pretty_str)
            pretty_rule = ""
            for pretty_cond in pretty_conds:
                if pretty_rule != "":
                    pretty_rule = pretty_rule + "<span style=\"color: Green; font-weight: bold;\"><br>AND<br></span>"
                pretty_rule = pretty_rule + pretty_cond
            pretty_rule = "<div style=\"line-height:140%; margin-top: 0px; margin-bottom: 0px;\">" + pretty_rule + \
                          "</div>"
            pretty_rules.append(pretty_rule)

        all_rules_html_text = ""
        for pretty_rule in pretty_rules:
            if all_rules_html_text != "":
                all_rules_html_text = all_rules_html_text + "<div style=\"color: DodgerBlue; font-weight: bold; " \
                                                            "margin-top: 5px; margin-bottom: 5px;\">&emsp;OR</div>"
            all_rules_html_text = all_rules_html_text + pretty_rule
        return all_rules_html_text

    def gen_rule_box_metrics(self):
        conf_matrix = self.gen_conf_matrix()
        avg_metrics = self.gen_avg_rule_metrics()
        metrics_box = VBox([conf_matrix, avg_metrics], layout=Layout(margin='35px 0px 0px 30px'))
        return metrics_box

    def gen_conf_matrix(self):
        header_color = "AliceBlue"
        cell_color = "Snow"
        font_size = '12px'
        html_rule_performance = HTML("<span style=\"font-weight: bold; font-size:16px\">Rule Performance</span>")
        conf_matrix = ipysheet.sheet(rows=4, columns=4, column_headers=False, row_headers=False)
        ipysheet.cell(0, 0, '', read_only=True, background_color=header_color)
        ipysheet.cell(0, 1, 'Rule = True', read_only=True,
                      style={'font-weight': 'bold', 'color': 'Green', 'font-size': font_size},
                      background_color=header_color)
        ipysheet.cell(0, 2, 'Rule = False', read_only=True,
                      style={'font-weight': 'bold', 'color': 'Red', 'font-size': font_size},
                      background_color=header_color)
        ipysheet.cell(0, 3, 'Covered by rule', read_only=True, style={'font-weight': 'bold', 'font-size': font_size},
                      background_color=header_color)
        ipysheet.cell(1, 0, 'High case duration', read_only=True, style={'font-weight': 'bold', 'font-size': font_size},
                      background_color=header_color)
        ipysheet.cell(1, 1, str(self.dr_miner.metrics['true_p']), read_only=True, style={'font-size': font_size},
                      background_color=cell_color)
        ipysheet.cell(1, 2, str(self.dr_miner.metrics['false_n']), read_only=True, style={'font-size': font_size},
                      background_color=cell_color)
        ipysheet.cell(1, 3, str(round(self.dr_miner.metrics['recall_p'] * 100)) + "%", read_only=True,
                      style={'font-size': font_size}, background_color=cell_color)
        ipysheet.cell(2, 0, 'Low case duration', read_only=True, style={'font-weight': 'bold', 'font-size': font_size},
                      background_color=header_color)
        ipysheet.cell(2, 1, str(self.dr_miner.metrics['false_p']), read_only=True, style={'font-size': font_size},
                      background_color=cell_color)
        ipysheet.cell(2, 2, str(self.dr_miner.metrics['true_n']), read_only=True, style={'font-size': font_size},
                      background_color=cell_color)
        ipysheet.cell(2, 3, str(round(self.dr_miner.metrics['recall_n'] * 100)) + "%", read_only=True,
                      style={'font-size': font_size}, background_color=cell_color)
        ipysheet.cell(3, 0, 'Rule correct', read_only=True, style={'font-weight': 'bold', 'font-size': font_size},
                      background_color=header_color)
        ipysheet.cell(3, 1, str(round(self.dr_miner.metrics['precision_p'] * 100)) + "%", read_only=True,
                      style={'font-size': font_size}, background_color=cell_color)
        ipysheet.cell(3, 2, str(round(self.dr_miner.metrics['precision_n'] * 100)) + "%", read_only=True,
                      style={'font-size': font_size}, background_color=cell_color)
        ipysheet.cell(3, 3, '', read_only=True, background_color=cell_color)
        vbox_all = VBox(children=[html_rule_performance, conf_matrix])
        return vbox_all

    def gen_avg_rule_metrics(self):
        html_avg_true = Box([HTML("<center><span style=\"font-weight:bold\"> Rule = True</span><br><span "
                                  "style=\"color: Red; font-size:16px\">" + str(
            round(self.dr_miner.metrics['avg_True'])) + "\xa0" + self.label_unit + "</span></center>")],
                            layout=Layout(border='3px double CornflowerBlue', margin='0px 10px 0px 0px'))
        html_avg_false = Box([HTML("<center><span style=\"font-weight:bold\"> Rule = False</span><br><span "
                                   "style=\"color: Green; font-size:16px\">" + str(
            round(self.dr_miner.metrics['avg_False'])) + "\xa0" + self.label_unit + "</span></center>")],
                             layout=Layout(border='3px double CornflowerBlue', color='CornflowerBlue',
                                           margin='0px 0px 0px 10px'))
        hbox_metrics = HBox([html_avg_true, html_avg_false])
        html_avg_case_duration = HTML("<span style=\"font-weight: bold; font-size:16px\">Average case duration</span>")
        vbox_metrics = VBox([html_avg_case_duration, hbox_metrics], layout=Layout(margin='10px 0px 0px 0px'))
        return vbox_metrics
