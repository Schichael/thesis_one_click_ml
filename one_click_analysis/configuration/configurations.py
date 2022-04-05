import abc
from typing import Any
from typing import Dict

from ipywidgets import Box
from ipywidgets import DatePicker
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox
from pycelonis.celonis_api.pql.pql import PQLFilter

from one_click_analysis import utils
from one_click_analysis.feature_processing.feature_processor import FeatureProcessor


class Configuration(abc.ABC):
    """Abstract class for a configuration"""

    def __init__(self, caption_size, caption_bold):
        self.caption_size = caption_size
        self.caption_bold = caption_bold

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

    @abc.abstractmethod
    def apply(self):
        """Define what happens when the configuration is applied."""
        pass


class DatePickerConfig(Configuration):
    """Configuration for defining a start and end date"""

    def __init__(
        self, fp: FeatureProcessor, caption_size: int = 14, caption_bold: bool = True
    ):
        """

        :param fp: FeatureProcessor before features were processes
        :param caption_size: size of the caption in pixels
        :param caption_bold: whether to have the caption in bold or not.
        """
        super().__init__(caption_size, caption_bold)

        self.fp = fp
        self._config = {}
        self._config_box = Box()
        self.datepicker_start = None
        self.datepicker_end = None
        self.create_config_box()

    @property
    def config(self):
        return self._config

    @property
    def config_box(self):
        return self._config_box

    @config_box.setter
    def config_box(self, value):
        self._config_box = value

    def get_html_str_caption_bold(self):
        if self.caption_bold:
            return "bold"
        else:
            return "normal"

    def apply(self):
        """Apply the configuration"""
        if "date_start" in self.config and self.config["date_start"] is not None:
            date_str_pql = (
                f"{{d'{utils.convert_date_to_str(self.config['date_start'])}'}}"
            )
            filter_str = (
                f'PU_FIRST("{self.fp.case_table_name}", '
                f'"{self.fp.activity_table_name}"."'
                f'{self.fp.eventtime_col}") >= {date_str_pql}'
            )
            self.fp.filters.append(PQLFilter(filter_str))
        if "date_end" in self.config and self.config["date_end"] is not None:
            date_str_pql = (
                f"{{d'{utils.convert_date_to_str(self.config['date_end'])}'}}"
            )
            filter_str = (
                f'PU_LAST("{self.fp.case_table_name}", '
                f'"{self.fp.activity_table_name}"."'
                f'{self.fp.eventtime_col}") <= {date_str_pql}'
            )
            self.fp.filters.append(PQLFilter(filter_str))

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

        def bind_datepicker_end(b):
            self.config["date_end"] = b.new

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
            f':{self.caption_size}px">Pick date interval for analysis</span>'
        )
        caption_HTML = HTML(html_caption_str)
        hbox_datepickers = HBox(children=[vbox_datepicker_start, vbox_datepicker_end])
        box_config = VBox(children=[caption_HTML, hbox_datepickers])
        self.config_box = box_config
