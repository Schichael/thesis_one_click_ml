from IPython.display import display
from ipywidgets import Tab
from ipywidgets import widgets

import utils
from DecisionRuleScreen import DecisionRulesBox
from feature_processing.feature_processor import FeatureProcessor
from gui.overview_screen import OverviewScreen
from gui.statistical_analysis_screen import StatisticalAnalysisBox


class AnalysisCaseDuration:
    def __init__(self, datamodel, celonis_login=None):
        self.datamodel = datamodel
        self.celonis_login = celonis_login
        self.fp = None
        self.df_total_time = None
        self.overview_box = None
        self.stat_analysis_box = None
        self.dec_rule_box = None
        self.tabs = None

    def run(self):
        out = widgets.Output(layout={"border": "1px solid black"})
        display(out)
        # Connect and get dm
        with out:
            print("Connecting to Celonis...")
        dm = utils.get_dm(self.datamodel, celonis_login=self.celonis_login)
        with out:
            print("Done")
        # Create dm_info and preprocessor
        with out:
            print("Fetching data and preprocessing...")
        self.dm_info, self.fp, self.df_total_time = self.preprocess(dm)
        with out:
            print("Done")

        with out:
            print("Creatng GUI...")
        # Create overview box
        overview_box_obj = OverviewScreen(self.fp)
        self.overview_box = overview_box_obj.get_overview_box()

        # Ceate statistical analysis tab
        stat_analysis_obj = StatisticalAnalysisBox(self.fp)
        self.stat_analysis_box = stat_analysis_obj.get_statistical_box()

        # Create decision rule miner box
        dec_rule_box_obj = DecisionRulesBox(
            self.fp.df,
            self.fp.label.df_attribute_name,
            self.fp.label.unit,
            self.fp.attributes_dict,
            pos_class=None,
        )
        self.dec_rule_box = dec_rule_box_obj.create_view()

        # Create tabs
        self.tabs = self.create_tabs()
        out.close()
        del out
        display(self.tabs)

    def preprocess(self, dm):
        fp = FeatureProcessor(dm)
        fp.run_total_time_PQL(20, time_aggregation="DAYS")
        return fp

    def create_tabs(self):
        tab_names = ["Overview", "Statistical Analysis", "Decision Rules"]
        tab = Tab([self.overview_box, self.stat_analysis_box, self.dec_rule_box])
        for i, el in enumerate(tab_names):
            tab.set_title(i, el)

        return tab
