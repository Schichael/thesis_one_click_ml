import abc
from typing import Any
from typing import Dict

from ipywidgets import Box
from ipywidgets import Checkbox
from ipywidgets import DatePicker
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import Select
from ipywidgets import VBox
from pycelonis.celonis_api.pql.pql import PQLFilter

from one_click_analysis import utils
from one_click_analysis.errors import ConfiguratorNotSetError
from one_click_analysis.feature_processing.feature_processor import FeatureProcessor


class Configuration(abc.ABC):
    """Abstract class for a configuration"""

    def __init__(
        self, required: bool = False, caption_size: int = 14, caption_bold: bool = True
    ):
        self.caption_size = caption_size
        self.caption_bold = caption_bold
        self.required = required
        self.optional_or_required_str = "required" if self.required else "optional"
        self._configurator = None

    def get_html_str_caption_bold(self):
        if self.caption_bold:
            return "bold"
        else:
            return "normal"

    @property
    @abc.abstractmethod
    def requirement_met(self) -> bool:
        """Check whether the requirement is met if self.required = True

        :return:
        """
        pass

    @property
    def configurator(self):
        if self._configurator is None:
            raise ConfiguratorNotSetError()
        else:
            return self._configurator

    @configurator.setter
    def configurator(self, value: "Configurator"):  # noqa
        self._configurator = value

    @property
    @abc.abstractmethod
    def config(self) -> Dict[str, Any]:
        """dictionary that holds the configuration"""
        pass

    @property
    @abc.abstractmethod
    def config_box(self):
        """ipywidgets Box to display the config visualization"""
        pass

    @abc.abstractmethod
    def create_config_box(self) -> Box:
        """Create box with the configuration

        :return: Box with the configuration
        """
        pass


class DatePickerConfig(Configuration):
    """Configuration for defining a start and end date"""

    def __init__(self, fp: FeatureProcessor, **kwargs):
        """

        :param fp: FeatureProcessor before features were processes
        :param caption_size: size of the caption in pixels
        :param caption_bold: whether to have the caption in bold or not.
        """
        super().__init__(**kwargs)

        self.fp = fp
        self._config = {}
        self._config_box = Box()
        self.datepicker_start = None
        self.datepicker_end = None
        self.create_config_box()

    @property
    def requirement_met(self):
        if not self.required:
            return True
        if self.datepicker_start is not None and self.datepicker_end is not None:
            return True
        else:
            return False

    @property
    def config(self):
        return self._config

    @property
    def config_box(self):
        return self._config_box

    @config_box.setter
    def config_box(self, value):
        self._config_box = value

    def create_filter_queries(self):
        filter_start = filter_end = None
        if "date_start" in self.config and self.config["date_start"] is not None:
            date_str_pql = (
                f"{{d'{utils.convert_date_to_str(self.config['date_start'])}'}}"
            )
            filter_str = (
                f'PU_FIRST("{self.fp.case_table_name}", '
                f'"{self.fp.activity_table_name}"."'
                f'{self.fp.eventtime_col}") >= {date_str_pql}'
            )
            filter_start = PQLFilter(filter_str)
        if "date_end" in self.config and self.config["date_end"] is not None:
            date_str_pql = (
                f"{{d'{utils.convert_date_to_str(self.config['date_end'])}'}}"
            )
            filter_str = (
                f'PU_LAST("{self.fp.case_table_name}", '
                f'"{self.fp.activity_table_name}"."'
                f'{self.fp.eventtime_col}") <= {date_str_pql}'
            )
            filter_end = PQLFilter(filter_str)
        filters = []
        if filter_start is not None:
            filters.append(filter_start)
        if filter_end is not None:
            filters.append(filter_end)
        return filters

    def create_config_box(self):
        """Create ipywidgets Box object for configuration visualization."""
        html_descr_datepicker_start = HTML(
            '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px; '
            'font-size: 14px;">Earliest Start Date</div>'
        )
        html_descr_datepicker_end = HTML(
            '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px; '
            'font-size: 14px;">Latest End Date</div>'
        )
        self.datepicker_start = DatePicker(disabled=False)
        self.datepicker_end = DatePicker(disabled=False)

        def bind_datepicker_start(b):
            self.config["date_start"] = b.new
            filters = self.create_filter_queries()
            self.configurator.filters[self] = filters

        def bind_datepicker_end(b):
            self.config["date_end"] = b.new
            filters = self.create_filter_queries()
            self.configurator.filters[self] = filters

        self.datepicker_start.observe(bind_datepicker_start, "value")
        self.datepicker_end.observe(bind_datepicker_end, "value")
        vbox_datepicker_start = VBox(
            children=[html_descr_datepicker_start, self.datepicker_start]
        )
        vbox_datepicker_end = VBox(
            children=[html_descr_datepicker_end, self.datepicker_end],
            layout=Layout(margin="0px 0px 0px 10px"),
        )
        html_caption_str = (
            f'<span style="font-weight:'
            f"{self.get_html_str_caption_bold()}; font-size"
            f':{self.caption_size}px">Pick date interval for analysis ('
            f"{self.optional_or_required_str})</span>"
        )
        caption_HTML = HTML(html_caption_str)
        hbox_datepickers = HBox(children=[vbox_datepicker_start, vbox_datepicker_end])
        box_config = VBox(children=[caption_HTML, hbox_datepickers])
        self.config_box = box_config


class TransitionConfig(Configuration):
    """Configuration for defining a source activity and target activities"""

    def __init__(self, fp: FeatureProcessor, **kwargs):
        """
        :param fp: FeatureProcessor before features were processes
        """
        super().__init__(**kwargs)

        self.fp = fp
        self._config = {}
        self._config_box = Box()
        self.selected_source_activity = None
        self.selected_target_activities = []
        self.create_config_box()

    @property
    def requirement_met(self):
        if not self.required:
            return True
        if (
            self.selected_source_activity is not None
            and self.selected_target_activities
        ):
            return True
        else:
            return False

    @property
    def config(self):
        return self._config

    @property
    def config_box(self):
        return self._config_box

    @config_box.setter
    def config_box(self, value):
        self._config_box = value

    def create_config_box(self):
        """Create ipywidgets Box object for configuration visualization."""
        html_descr_source_activity = HTML(
            '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px; '
            'font-size: 14px;">Pick a source activity</div>'
        )
        html_descr_target_activities = HTML(
            '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px; '
            'font-size: 14px;">Pick target activities</div>'
        )

        activities = self.fp.get_activities()["activity"].values
        # Sort activities
        activities = sorted(activities)

        def on_source_activity_clicked(b):
            self.selected_source_activity = b.new
            self.config["source_activity"] = self.selected_source_activity

        # Source Activity
        source_activity_selection = Select(
            options=activities,
            value=None,
            layout=Layout(overflow="auto", height="auto", max_height="400px"),
        )
        source_activity_selection.observe(on_source_activity_clicked, "value")
        vbox_source_activity_selection = VBox(
            children=[html_descr_source_activity, source_activity_selection]
        )

        # Target Activities
        def on_checkbox_clicked(b):
            """Define behaviour when checkbox of a "normal" attribute (not activity
            or case column attribute) is toggled

            :param b:
            :return:
            """
            if b.new is False:
                self.selected_target_activities.remove(b.owner.description)
            else:
                self.selected_target_activities.append(b.owner.description)

            self.config["target_activities"] = self.selected_target_activities

        checkboxes = []
        for activity in activities:
            cb = Checkbox(value=False, description=activity, indent=False)
            cb.observe(on_checkbox_clicked, "value")
            checkboxes.append(cb)

        vbox_target_activities_cbs = VBox(
            children=checkboxes,
            layout=Layout(overflow="auto", max_height="400px"),
        )

        vbox_target_activities = VBox(
            children=[html_descr_target_activities, vbox_target_activities_cbs]
        )

        html_caption_str = (
            f'<span style="font-weight:'
            f"{self.get_html_str_caption_bold()}; font-size"
            f':{self.caption_size}px">Pick transition activities for analysis '
            f"({self.optional_or_required_str})</span>"
        )
        caption_HTML = HTML(html_caption_str)
        hbox_activity_selection = HBox(
            children=[vbox_source_activity_selection, vbox_target_activities]
        )
        box_config = VBox(children=[caption_HTML, hbox_activity_selection])
        self.config_box = box_config
