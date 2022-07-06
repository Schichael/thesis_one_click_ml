from typing import Any
from typing import List

from IPython.display import display
from ipywidgets import HTML
from ipywidgets import Tab
from ipywidgets import VBox
from ipywidgets import widgets

from one_click_analysis.configuration.configurations import ActivityTableConfig
from one_click_analysis.configuration.configurations import ConformanceQueryConfig
from one_click_analysis.configuration.configurations import DatamodelConfig
from one_click_analysis.configuration.configurations import DatePickerConfig
from one_click_analysis.configuration.configurations import IsClosedConfig
from one_click_analysis.configuration.configurator import Configurator
from one_click_analysis.configuration.configurator_view import ConfiguratorView
from one_click_analysis.gui.description_screen import DescriptionScreen
from one_click_analysis.violation_processing.gui.violation_selection import (
    ViolationSelectionScreen,
)
from one_click_analysis.violation_processing.violation import Violation
from one_click_analysis.violation_processing.violation import ViolationType
from one_click_analysis.violation_processing.violation_processor import (
    ViolationProcessor,
)


class AnalysisViolation:
    """Analysis of potential effects on case duration."""

    def __init__(self, login=None):
        """

        :param datamodel: datamodel name or id
        :param celonis_login: dict with login information
        """

        self.activity_table_str = None
        self.datamodel = None
        self.login = login
        self.dm = None
        self.process_config = None
        self.case_duration_processor = None
        self.df_total_time = None
        self.configurator = None
        self.description_view = None
        self.config_view = None
        self.overview_screen = None
        self.stat_analysis_screen = None
        self.dec_rule_screen = None
        self.expert_screen = None
        self.attr_selection = None
        self.tabs = None
        self.tab_names = [
            "Description",
            "Configurations",
            "Violations",
            "Analysis",
        ]
        self.time_unit = "DAYS"

        self.selected_attributes = []
        self.selected_activity_table_cols = []
        self.selected_case_table_cols = []

    def _create_description(self):

        name_str = "Case duration Analysis"
        goal_str = "Goals here."
        definition_str = "Description here."

        self.description_view = DescriptionScreen(
            analysis_name=name_str,
            analysis_goal=goal_str,
            analysis_definition=definition_str,
            static_attribute_descriptors=[],
            dynamic_attribute_descriptors=[],
        )
        self.description_view.create_description_screen()

    def _create_config(self):
        """Create config view.
        The analysis needs the following configs:
        DatamodelConfig
        ActivityTableConfig
        DatePickerConfig
        AttributeSelectionConfig

        :return:
        """
        self.configurator = Configurator()
        config_dm = DatamodelConfig(
            configurator=self.configurator, celonis_login=self.login, required=True
        )
        config_activity_table = ActivityTableConfig(
            configurator=self.configurator,
            datamodel_identifier="datamodel",
            required=True,
        )

        config_closed_cases = IsClosedConfig(
            configurator=self.configurator,
            datamodel_identifier="datamodel",
            activity_table_identifier="activity_table",
            required=False,
            additional_prerequsit_config_ids=[],
        )

        config_datepicker = DatePickerConfig(
            configurator=self.configurator,
            datamodel_identifier="datamodel",
            activity_table_identifier="activity_table",
            required=False,
            additional_prerequsit_config_ids=["is_closed"],
        )

        config_conformance_query = ConformanceQueryConfig(
            configurator=self.configurator,
            config_identifier="conformance_query",
            required=True,
            additional_prerequsit_config_ids=["datepicker"],
        )

        # Set the subsequrnt configurations that are updated when the respective
        # configuration is applied or updated itself
        config_dm.subsequent_configurations = [config_activity_table]
        config_activity_table.subsequent_configurations = [config_closed_cases]
        config_closed_cases.subsequent_configurations = [config_datepicker]
        config_datepicker.subsequent_configurations = [config_conformance_query]
        config_conformance_query.subsequent_configurations = []
        self.config_view = ConfiguratorView(
            configurations=[
                config_dm,
                config_activity_table,
                config_closed_cases,
                config_datepicker,
                config_conformance_query,
            ],
            run_analysis=self.run_analysis,
        )

    def run(self):
        # 1. Connect to Celonis and get dm
        self._create_description()
        self._create_config()

        # 2. Create FeatureProcessor and Configurator
        # self.process_config =

        self.tabs = self.create_tabs(
            [
                self.description_view.description_box,
                self.config_view.configurator_box,
                widgets.VBox(),
                widgets.VBox(),
            ]
        )
        display(self.tabs)

    def run_analysis(self):
        # Reset fp from a previous run

        # Get configurations
        # datepicker_configs = self.configurator.config_dict.get("datepicker")

        self.process_config = self.configurator.config_dict["datamodel"][
            "process_config"
        ]
        activity_table_str = self.configurator.config_dict["activity_table"][
            "activity_table_str"
        ]
        is_closed_query = self.configurator.config_dict["is_closed"]["pql_query"]
        conformance_query_str = self.configurator.config_dict["conformance_query"][
            "conformance_query"
        ]
        filters_datepicker = self.configurator.filter_dict.get("datepicker")
        if filters_datepicker is None:
            filters_datepicker = []
        self.violation_processor = ViolationProcessor(
            conformance_query_str=conformance_query_str,
            process_config=self.process_config,
            activity_table_str=activity_table_str,
            filters=filters_datepicker,
            is_closed_query_str=is_closed_query.query,
        )

        self.violation_selection_screen = ViolationSelectionScreen(
            violations=self.violation_processor.violations,
            violation_df=self.violation_processor.violations_df,
            timestamp_column=self.violation_processor.timestamp_column,
            time_aggregation=self.time_unit,
            configurator=self.configurator,
            update_tab_function=self.update_analysis_tab,
        )

        self.violation_selection_screen.create_violation_selection_screen()

        # Create expert box
        # attributes = self.case_duration_processor.static_attributes +
        # self.case_duration_processor.dynamic_attributes

        # self.expert_screen = ExpertScreen(
        #    attributes=attributes,
        #    activity_table_cols=self.fp.dynamic_categorical_cols
        #    + self.fp.dynamic_numerical_cols,
        #    case_table_cols={
        #        "table name": self.fp.static_categorical_cols
        #        + self.fp.static_numerical_cols
        #    },
        #    features=self.fp.features,
        #    attr_selection=self.attr_selection,
        # )
        # self.expert_screen.create_expert_box()

        # Create tabs
        self.update_tabs(
            [
                self.description_view.description_box,
                self.config_view.configurator_box,
                self.violation_selection_screen.violation_selection_box,
                VBox(),
            ]
        )
        # out.close()
        # del out
        # display(self.tabs)

    def create_tabs(self, tab_contents: List[widgets.widget.Widget]):
        """Create the tabs for the GUI.

        :return:
        """
        tab = Tab(tab_contents)
        for i, el in enumerate(self.tab_names):
            tab.set_title(i, el)

        return tab

    def update_tabs(self, tab_contents: List[widgets.widget.Widget]):
        self.tabs.children = tab_contents

    def update_analysis_tab(self, content: Any, violation: Violation):
        """Update the analysis tab"""
        if violation.violation_type == ViolationType.INCOMPLETE:
            violation_str = "Incomplete case"
        else:
            violation_str = violation.violation_readable

        headline_analysis_tab_str = (
            f'<div style="line-height:140%;font-weight:bold; font-size: 16px'
            f'">'
            f"Violation: {violation_str}</div>"
        )
        html_headline = HTML(headline_analysis_tab_str)
        analysis_tab = VBox(children=[html_headline, content])
        self.tabs.children = [
            self.description_view.description_box,
            self.config_view.configurator_box,
            self.violation_selection_screen.violation_selection_box,
            analysis_tab,
        ]
