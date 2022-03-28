from typing import Optional

from IPython.display import display
from ipywidgets import Tab
from ipywidgets import widgets

import utils
from feature_processing.feature_processor import FeatureProcessor
from gui.decision_rule_screen import DecisionRulesScreen
from gui.overview_screen import OverviewScreen
from gui.statistical_analysis_screen import StatisticalAnalysisBox


class AnalysisCaseDuration:
    """Analysis of potential effects on case duration."""

    def __init__(self, datamodel: str, celonis_login: Optional[dict] = None):
        """

        :param datamodel: datamodel name or id
        :param celonis_login: dict with login information
        """

        self.datamodel = datamodel
        self.celonis_login = celonis_login
        self.fp = None
        self.df_total_time = None
        self.overview_box = None
        self.stat_analysis_box = None
        self.dec_rule_box = None
        self.tabs = None

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
        self.fp.run_total_time_PQL(20, time_aggregation="DAYS")
        with out:
            print("Done")

        # 3. Create the GUI
        with out:
            print("Creatng GUI...")
        # Create overview box
        overview_box_obj = OverviewScreen(self.fp)
        self.overview_box = overview_box_obj.get_overview_screen()

        # Ceate statistical analysis tab
        stat_analysis_obj = StatisticalAnalysisBox(self.fp)
        self.stat_analysis_box = stat_analysis_obj.create_statistical_screen()

        # Create decision rule miner box
        dec_rule_box_obj = DecisionRulesScreen(
            self.fp,
            pos_class=None,
        )
        self.dec_rule_box = dec_rule_box_obj.create_decision_rule_screen()

        # Create tabs
        self.tabs = self.create_tabs()
        out.close()
        del out
        display(self.tabs)

    def create_tabs(self):
        """Create the tabs for the GUI.

        :return:
        """
        tab_names = ["Overview", "Statistical Analysis", "Decision Rules"]
        tab = Tab([self.overview_box, self.stat_analysis_box, self.dec_rule_box])
        for i, el in enumerate(tab_names):
            tab.set_title(i, el)

        return tab
