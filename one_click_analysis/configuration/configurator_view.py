from typing import Callable
from typing import List

from ipywidgets import Button
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

from one_click_analysis.configuration.configurations import Configuration


class ConfiguratorView:
    """Class that creates a screen with the configurations and holds the configurations.
    When the "Run analysis" button is clicked and it was run before, the fp is
    first reset to its initial values
    """

    def __init__(
        self,
        configurations: List[Configuration],
        run_analysis: Callable,
        **run_analysis_args,
    ):
        """

        :param configurations: List with configuration objects
        :param run_analysis: function to call when the apply button is clicked
        :param run_analysis_args: arguments for the run_analysis function
        """
        self.configurations = configurations
        self.run_analysis = run_analysis
        self.run_analysis_args = run_analysis_args
        self.children_config_box_layouts = Layout(margin="20px 0px 0px 0px")
        # Dictionary with filter queries. They keys are the configuration instances,
        # the values a list of PQL Filter queries
        self.filters = {}
        self.apply_button = None
        # self._set_layouts_configs()
        self._init_configurations()
        self.configurator_box = self.create_box()
        self.applied_configs = {}  # Configs at the time the apply button was clicked
        self.button_was_clicked = False

    def _set_layouts_configs(self):
        for config in self.configurations:
            config.config_box.layout = self.children_config_box_layouts

    def _init_configurations(self):
        """Give the view's update function to the configurations"""
        for config in self.configurations:
            config.set_configurator_view_update_fct(self.update_view)

    def on_apply_clicked(self, b):
        """Define what happens when apply button is clicked"""
        # Reset applied configs
        for config in self.configurations:
            if not config.configurator_requirement_met:
                return

        self.apply_button.disabled = True

        self.run_analysis(**self.run_analysis_args)
        self.button_was_clicked = True
        self.apply_button.disabled = False

    def update_view(self):
        html_title = HTML(
            '<span style="font-weight:bold;  font-size:16px">Configurations</span'
        )
        self._set_layouts_configs()
        self.configurator_box.children = (
            [html_title]
            + [config.config_box for config in self.configurations]
            + [self.apply_button]
        )

    def create_box(self) -> VBox:
        """Create ipywidgets box with the configuration selections"""
        self._set_layouts_configs()
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
