from typing import List
from typing import Optional

from IPython.display import display
from ipywidgets import Tab
from ipywidgets import widgets

from one_click_analysis import utils
from one_click_analysis.attribute_selection import AttributeSelection
from one_click_analysis.feature_processing import attributes
from one_click_analysis.feature_processing.feature_processor import FeatureProcessor
from one_click_analysis.gui.decision_rule_screen import DecisionRulesScreen
from one_click_analysis.gui.expert_screen import ExpertScreen
from one_click_analysis.gui.overview_screen import OverviewScreen
from one_click_analysis.gui.statistical_analysis_screen import StatisticalAnalysisScreen


class AttributeSelectionCaseDuration(AttributeSelection):
    def __init__(
        self,
        selected_attributes: List[attributes.MinorAttribute],
        selected_activity_table_cols: List[str],
        selected_case_table_cols: List[str],
        statistical_analysis_screen: StatisticalAnalysisScreen,
        decision_rules_screen: DecisionRulesScreen,
    ):
        super().__init__(
            selected_attributes, selected_activity_table_cols, selected_case_table_cols
        )
        self.statistical_analysis_screen = statistical_analysis_screen
        self.decision_rules_screen = decision_rules_screen

    def update(self):
        self.statistical_analysis_screen.update_attr_selection(
            self.selected_attributes,
            self.selected_activity_table_cols,
            self.selected_case_table_cols,
        )
        self.decision_rules_screen.update_attr_selection(
            self.selected_attributes,
            self.selected_activity_table_cols,
            self.selected_case_table_cols,
        )


class AnalysisCaseDuration:
    """Analysis of potential effects on case duration."""

    def __init__(
        self, datamodel: str, celonis_login: Optional[dict] = None, th: float = 0.3
    ):
        """

        :param datamodel: datamodel name or id
        :param celonis_login: dict with login information
        """

        self.datamodel = datamodel
        self.celonis_login = celonis_login
        self.th = th
        self.fp = None
        self.df_total_time = None
        self.overview_screen = None
        self.stat_analysis_screen = None
        self.dec_rule_screen = None
        self.expert_screen = None
        self.tabs = None

        self.selected_attributes = []
        self.selected_activity_table_cols = []
        self.selected_case_table_cols = []

    def run(self):
        """Run the pipeline to create the analysis.
        1. Connect to Celonis and get dm
        2. Create FeatureProcessor
        3. Create the GUI

        :return:
        """
        out = widgets.Output(layout={"border": "1px solid black"})
        display(out)
        # 1. Connect to Celonis and get dm
        with out:
            print("Connecting to Celonis...")
        dm = utils.get_dm(self.datamodel, celonis_login=self.celonis_login)
        with out:
            print("Done")
        # 2. Create FeatureProcessor and run processing for the analysis
        with out:
            print("Fetching data and preprocessing...")
        self.fp = FeatureProcessor(dm)
        self.fp.run_total_time_PQL(time_unit="DAYS")
        with out:
            print("Done")

        # assign the attributes and columns
        self.selected_attributes = self.fp.minor_attrs
        self.selected_activity_table_cols = (
            self.fp.dynamic_categorical_cols + self.fp.dynamic_numerical_cols
        )
        self.selected_case_table_cols = (
            self.fp.static_categorical_cols + self.fp.static_numerical_cols
        )

        # 3. Create the GUI
        with out:
            print("Creatng GUI...")
        # Create overview box
        self.overview_screen = OverviewScreen(self.fp)
        self.overview_screen.create_overview_screen()

        # Ceate statistical analysis tab
        self.stat_analysis_screen = StatisticalAnalysisScreen(
            self.fp,
            self.th,
            self.selected_attributes,
            self.selected_activity_table_cols,
            self.selected_case_table_cols,
        )
        self.stat_analysis_screen.create_statistical_screen()

        # Create decision rule miner box
        self.dec_rule_screen = DecisionRulesScreen(
            self.fp,
            self.selected_attributes,
            self.selected_activity_table_cols,
            self.selected_case_table_cols,
            pos_class=None,
        )
        self.dec_rule_screen.create_decision_rule_screen()

        # Create AttributeSelection object
        attr_selection_case_duration = AttributeSelectionCaseDuration(
            self.selected_attributes,
            self.selected_activity_table_cols,
            self.selected_case_table_cols,
            self.stat_analysis_screen,
            self.dec_rule_screen,
        )

        # Create expert box
        self.expert_screen = ExpertScreen(self.fp, attr_selection_case_duration)
        self.expert_screen.create_expert_box()

        # Create tabs
        self.tabs = self.create_tabs()
        out.close()
        del out
        display(self.tabs)

    def create_tabs(self):
        """Create the tabs for the GUI.

        :return:
        """
        tab_names = [
            "Overview",
            "Statistical Analysis",
            "Decision Rules",
            "Expert Tab",
        ]
        tab = Tab(
            [
                self.overview_screen.overview_box,
                self.stat_analysis_screen.statistical_analysis_box,
                self.dec_rule_screen.decision_rule_box,
                self.expert_screen.expert_box,
            ]
        )
        for i, el in enumerate(tab_names):
            tab.set_title(i, el)

        return tab
