from typing import Callable
from typing import List

from ipywidgets import Button
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

from one_click_analysis.configuration.configurations import Configuration
from one_click_analysis.feature_processing.feature_processor import FeatureProcessor


class Configurator:
    """Class that creates a screen with the configurations and holds the configurations.
    When the "Run analysis" button is clicked and it was run before, the fp is
    first reset to its initial values
    """

    def __init__(
        self,
        fp: FeatureProcessor,
        configurations: List[Configuration],
        run_analysis: Callable,
        *run_analysis_args
    ):
        """

        :param configurations: List with configuration objects
        :param run_analysis: function to call when the apply button is clicked
        :param run_analysis_args: arguments for the run_analysis function
        """
        self.fp = fp
        self.configurations = configurations
        self.run_analysis = run_analysis
        self.run_analysis_args = run_analysis_args
        self.children_config_box_layouts = Layout(margin="20px 0px 0px 0px")
        self.apply_button = None
        self._set_layouts_configs()
        self.configurator_box = self.create_box()
        self.applied_configs = {}  # Configs at the time the apply button was clicked
        self.button_was_clicked = False

    def _set_layouts_configs(self):
        for config in self.configurations:
            config.config_box.layout = self.children_config_box_layouts

    def on_apply_clicked(self, b):
        """Define what happens when apply button is clicked"""

        self.apply_button.disabled = True

        if self.button_was_clicked:
            self.fp.reset_fp()
        for config in self.configurations:
            config.apply()
            self.applied_configs.update(config.config)

        self.run_analysis(*self.run_analysis_args)
        self.button_was_clicked = True
        self.apply_button.disabled = False

    def create_box(self) -> VBox:
        """Create ipywidgets box with the configuration selections"""
        html_title = HTML(
            '<span style="font-weight:bold;  font-size:16px">Configurations</span'
        )
        self.apply_button = Button(
            description="Run Analysis", layout=self.children_config_box_layouts
        )
        self.apply_button.on_click(self.on_apply_clicked)
        vbox_children = (
            [html_title]
            + [config.config_box for config in self.configurations]
            + [self.apply_button]
        )
        return VBox(children=vbox_children)
