import abc
from typing import Any, Optional, List, Callable, Tuple
from typing import Dict

from ipywidgets import Box, widgets
from ipywidgets import Checkbox
from ipywidgets import DatePicker
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import Select
from ipywidgets import VBox
from pycelonis import get_celonis
from pycelonis.celonis_api.errors import PyCelonisNotFoundError
from pycelonis.celonis_api.pql.pql import PQLFilter

from one_click_analysis import utils
from one_click_analysis.configuration.configurator_class import Configurator
from one_click_analysis.errors import ConfiguratorNotSetError
from one_click_analysis.feature_processing.feature_processor import (
    FeatureProcessor,
)
from one_click_analysis.process_config import process_config_utils
from one_click_analysis.process_config.process_config import ProcessConfig


class Configuration(abc.ABC):
    """Abstract class for a configuration"""

    def __init__(
        self,
        configurator: Configurator,
        config_identifier: str,
        title: str,
        required: bool = False,
        caption_size: int = 14,
        caption_bold: bool = True,
    ):
        self.configurator = configurator

        self.config_identifier = config_identifier
        self.caption_size = caption_size
        self.caption_bold = caption_bold
        self.required = required
        self.optional_or_required_str = "required" if self.required else "optional"
        # The configurations that depend on this configuration and get updated when
        # this configuration is applied.
        self.subsequent_configurations = []
        self.config_box = Box()
        self.config: Dict[str, Any] = {}
        self.filters: List[PQLFilter] = []
        self.html_caption_str = self.set_html_title_str(title)
        self.configurator_view_update_fct: Optional[Callable] = None

    def set_configurator_view_update_fct(self, fct: Callable):
        """Set the update button of the configurator view"""
        self.configurator_view_update_fct = fct

    def set_html_title_str(self, title):
        html_caption_str = (
            f'<span style="font-weight:'
            f"{self.get_html_str_caption_bold()}; font-size"
            f':{self.caption_size}px">{title} ('
            f"{self.optional_or_required_str})</span>"
        )
        return html_caption_str

    def reset(self):
        """Remove configs and filters from this configuration from the configurator.
        Also reset any variables that need to be reset."""
        # Remove from config_dict
        if self.config_identifier in self.configurator.config_dict:
            self.configurator.config_dict.pop(self.config_identifier)
        # Remove from filter_dict
        if self.config_identifier in self.configurator.filter_dict:
            self.configurator.filter_dict.pop(self.config_identifier)
        # Reset config and filters
        self.config = {}
        self.filters = []

        # Reset local variables
        self.reset_local()

    def get_html_str_caption_bold(self):
        if self.caption_bold:
            return "bold"
        else:
            return "normal"

    def update(self):
        """Update this and the subsequent configurations. This is done in the
        following steps:
        1. Reset all subsequent configurations
        2. Write current configs and filters to the configurator if they are defined
        (e.g. by a selection)
        3. Create the config box
        4. Update the subsequent configurations
        """
        # Reset all subsequent configurations
        self.reset_local()
        self.reset()
        self._create_config_box()

        for sub_config in self.subsequent_configurations:
            sub_config.reset_local()
            sub_config.reset()
            sub_config._create_config_box()

        # Create box
        self._create_config_box()

        # Apply config in case there is a default configuration
        # self.apply()

    def _create_config_box(self):
        """Create config box. If the prerequisites are met, the true config box is
        created, else the placeholder box"""
        if self.validate_prerequisites():
            self._create_true_config_box()
        else:
            self._create_placeholder_config_box()

    def apply(self):
        """Define what happens when a configuration is applied. (E.g. by clicking on
        an "Apply" button). It must also work when the user has not made any
        configurations yet. It should write the configs and filters to the
        configurator and call the update methods subsequent configurations"""
        self.update_configurator()
        for sub_config in self.subsequent_configurations:
            sub_config.update()

        self.configurator_view_update_fct()

    def update_configurator(self):
        """Update the member variable of the configurator for the current
        configuration."""
        if not self.validate_prerequisites():
            return
        self.configurator.config_dict[self.config_identifier] = self.config
        self.configurator.filter_dict[self.config_identifier] = self.filters

    @property
    @abc.abstractmethod
    def requirement_met(self) -> bool:
        """Check whether the requirement is met if self.required = True. If
        self.required = True, the configuration must be somehow done by the user. If
        that was done, True is returned, else False. If False,
        the analysis/prediction shall not be allowed to be started yet.
        :return:
        """
        pass

    def reset_local(self):
        """Reset local variables if there are any that need to be reset when the
        configuration shall be reset. The variables self.config and self.filters are
        already reset in self.reset()"""
        pass

    # @property
    # def config(self) -> Dict[str, Any]:
    #    """dictionary that holds the configuration"""
    #    return self._config

    # @config.setter
    # def config(self, new_config: Dict[str, Any]):
    #    self._config = new_config

    # @property
    # def config_box(self):
    #    """ipywidgets Box to display the config visualization"""
    #    return self._config_box

    # @config_box.setter
    # def config_box(self, new_config_box: Any):
    #    self._config_box = new_config_box

    # @property
    # def subsequent_configurations(self):
    #    """List with subsequent configurations which update() and reset() methods
    #    are called from this configuration"""
    #    return self._subsequent_configurations

    # @subsequent_configurations.setter
    # def subsequent_configurations(self, function_list: List[Callable]):
    #    self._subsequent_configurations = function_list

    @abc.abstractmethod
    def _create_true_config_box(self) -> Box:
        """Create box with the configuration

        :return: Box with the configuration
        """
        pass

    def _create_placeholder_config_box(self):
        """Create placeholder box with the configuration when the real config box
        cannot be created yet.

        :return: Box with the placeholder configuration
        """
        caption_HTML = HTML(self.html_caption_str)
        hbox_datepickers = HBox()
        box_config = VBox(children=[caption_HTML, hbox_datepickers])
        self.config_box = box_config

    @abc.abstractmethod
    def validate_prerequisites(self) -> bool:
        """Validate if the prerequisites are met such that the configuration can be
        created. I.e. the configurator object already has the configs that this
        configuration needs to be created."""
        pass


class DatePickerConfig(Configuration):
    """Configuration for defining a start and end date"""

    def __init__(
        self,
        configurator: Configurator,
        datamodel_identifier: str,
        activity_table_identifier: str,
        config_identifier: str = "datepicker",
        **kwargs,
    ):
        """
        :param configurator: Configurator object
        :param datamodel_identifier: identifier of the datamodel in the configurator
        variables.
        :param config_identifier: Identifier of the datepicker config to be used in
        the configurator variables.
        """
        if "title" in kwargs:
            title = kwargs["title"]
        else:
            title = f"Pick date interval for analysis"
        super().__init__(
            configurator=configurator,
            config_identifier=config_identifier,
            title=title,
            **kwargs,
        )
        self.datamodel_identifier = datamodel_identifier
        self.activity_table_identifier = activity_table_identifier
        self.datepicker_start = None
        self.datepicker_end = None
        # Initialize config box
        self._create_config_box()

    @property
    def requirement_met(self):
        if not self.required:
            return True
        if self.datepicker_start is not None and self.datepicker_end is not None:
            return True
        else:
            return False

    def validate_prerequisites(self) -> bool:

        process_config_set = (
            self.datamodel_identifier in self.configurator.config_dict
            and self.configurator.config_dict[self.datamodel_identifier] is not None
        )
        activity_table_set = (
            self.activity_table_identifier in self.configurator.config_dict
            and self.configurator.config_dict[self.activity_table_identifier]
            is not None
        )

        return process_config_set and activity_table_set

    def create_filter_queries(self):
        filter_start = filter_end = None
        process_config = self.configurator.config_dict[self.datamodel_identifier][
            "process_config"
        ]
        activity_table_str = self.configurator.config_dict[
            self.activity_table_identifier
        ]["activity_table_str"]
        activity_table = process_config.table_dict[activity_table_str]

        if "date_start" in self.config and self.config["date_start"] is not None:
            date_str_pql = (
                f"{{d'{utils.convert_date_to_str(self.config['date_start'])}'}}"
            )
            filter_str = (
                f'PU_FIRST("{activity_table.case_table_str}", '
                f'"{activity_table.table_str}"."'
                f'{activity_table.eventtime_col_str}") >= {date_str_pql}'
            )
            filter_start = PQLFilter(filter_str)
        if "date_end" in self.config and self.config["date_end"] is not None:
            date_str_pql = (
                f"{{d'{utils.convert_date_to_str(self.config['date_end'])}'}}"
            )
            filter_str = (
                f'PU_FIRST("{activity_table.case_table_str}", '
                f'"{activity_table.table_str}"."'
                f'{activity_table.eventtime_col_str}") <= {date_str_pql}'
            )
            filter_end = PQLFilter(filter_str)

        filters = []
        if filter_start is not None:
            filters.append(filter_start)
        if filter_end is not None:
            filters.append(filter_end)
        return filters

    def _create_true_config_box(self):
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
            self.filters = self.create_filter_queries()
            self.apply()

        def bind_datepicker_end(b):
            self.config["date_end"] = b.new
            self.filters = self.create_filter_queries()
            self.apply()

        self.datepicker_start.observe(bind_datepicker_start, "value")
        self.datepicker_end.observe(bind_datepicker_end, "value")
        vbox_datepicker_start = VBox(
            children=[html_descr_datepicker_start, self.datepicker_start]
        )
        vbox_datepicker_end = VBox(
            children=[html_descr_datepicker_end, self.datepicker_end],
            layout=Layout(margin="0px 0px 0px 10px"),
        )

        caption_HTML = HTML(self.html_caption_str)
        hbox_datepickers = HBox(children=[vbox_datepicker_start, vbox_datepicker_end])
        box_config = VBox(children=[caption_HTML, hbox_datepickers])
        self.config_box = box_config


class DatamodelConfig(Configuration):
    """Configuration for selecting the datamodel. The datamodel is stored in config[
    'datamodel']. The process_config object is stored in config['process_config']"""

    def __init__(
        self,
        configurator: Configurator,
        config_identifier: str = "datamodel",
        celonis_login: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        if "title" in kwargs:
            title = kwargs["title"]
            kwargs.pop("title")
        else:
            title = f"Select a datamodel"

        super().__init__(
            configurator=configurator,
            config_identifier=config_identifier,
            title=title,
            **kwargs,
        )
        self.celonis_login = celonis_login
        self.celonis = self._get_celonis()
        # Initialize config box
        self._create_config_box()

    def _get_celonis(self):
        if self.celonis_login is None:
            celonis = get_celonis()
        else:
            celonis = get_celonis(**self.celonis_login)
        return celonis

    def _create_datamodel_list(self) -> Tuple[List[str], List[str]]:
        """Get List of the available datamodel names and their ids"""
        datamodels = self.celonis.datamodels
        datamodel_name_list = [dm.name for dm in datamodels]
        datamodel_id_list = [dm.id for dm in datamodels]
        return datamodel_name_list, datamodel_id_list

    def _create_true_config_box(self):
        text_field = widgets.Text(
            placeholder="Insert Datamodel ID", description="Datamodel ID:"
        )

        apply_button = widgets.Button(description="Apply")

        def on_apply_clicked(b):
            text_str = text_field.value
            try:
                dm = self.celonis.datamodels.find(text_str)
                self.config["datamodel"] = dm
                self.config["process_config"] = ProcessConfig(datamodel=dm)
                self.apply()
            except PyCelonisNotFoundError:
                return

        apply_button.on_click(on_apply_clicked)

        caption_HTML = HTML(self.html_caption_str)
        box_config = VBox(children=[caption_HTML, text_field, apply_button])
        self.config_box = box_config

    @property
    def requirement_met(self) -> bool:
        try:
            if (
                self.configurator.config_dict[self.config_identifier]["datamodel"]
                is not None
            ):
                return True
            else:
                return False
        except KeyError:
            # Entry in configurator not set yet
            return False

    def validate_prerequisites(self) -> bool:
        # Does not need any prerequisites
        return True


class ActivityTableConfig(Configuration):
    """Configuration for selecting the activity table. The name of the activity
    table is stored in config['activity_table_str']
    """

    def __init__(
        self,
        configurator: Configurator,
        datamodel_identifier: str,
        config_identifier: str = "activity_table",
        **kwargs,
    ):
        if "title" in kwargs:
            title = kwargs["title"]
            kwargs.pop("title")
        else:
            title = f"Select an activity table"

        super().__init__(
            configurator=configurator,
            config_identifier=config_identifier,
            title=title,
            **kwargs,
        )

        self.datamodel_identifier = datamodel_identifier

        # Initialize config box
        self._create_config_box()

    def _create_true_config_box(self):
        process_config = self.configurator.config_dict[self.datamodel_identifier][
            "process_config"
        ]
        activity_table_str_list = [
            table.table_str for table in process_config.activity_tables
        ]
        dropdown = widgets.Dropdown(
            options=activity_table_str_list, description="Select activity table:"
        )

        apply_button = widgets.Button(description="Apply")

        def on_apply_clicked(b):
            if dropdown.value is None:
                return
            self.config["activity_table_str"] = dropdown.value
            self.apply()

        apply_button.on_click(on_apply_clicked)

        caption_HTML = HTML(self.html_caption_str)
        box_config = VBox(children=[caption_HTML, dropdown, apply_button])
        self.config_box = box_config

    @property
    def requirement_met(self) -> bool:
        try:
            if (
                self.configurator.config_dict[self.config_identifier][
                    "activity_table_str"
                ]
                is not None
            ):
                return True
            else:
                return False
        except KeyError:
            # Entry in configurator not set yet
            return False

    def validate_prerequisites(self) -> bool:
        process_config_set = (
            self.datamodel_identifier in self.configurator.config_dict
            and self.configurator.config_dict[self.datamodel_identifier] is not None
        )
        return process_config_set


class DecisionConfig(Configuration):
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
            f':{self.caption_size}px">Pick transition activities for '
            f"analysis "
            f"({self.optional_or_required_str})</span>"
        )
        caption_HTML = HTML(html_caption_str)
        hbox_activity_selection = HBox(
            children=[vbox_source_activity_selection, vbox_target_activities]
        )
        box_config = VBox(children=[caption_HTML, hbox_activity_selection])
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
        self.selected_target_activity = None
        self.create_config_box()

    @property
    def requirement_met(self):
        if not self.required:
            return True
        if self.selected_source_activity is not None and self.selected_target_activity:
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
        html_descr_target_activity = HTML(
            '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px; '
            'font-size: 14px;">Pick a target activity</div>'
        )
        # TODO: Can get activities from the process_model directly
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

        def on_target_activity_clicked(b):
            self.selected_target_activity = b.new
            self.config["target_activity"] = self.selected_target_activity

        # Source Activity
        target_activity_selection = Select(
            options=activities,
            value=None,
            layout=Layout(overflow="auto", height="auto", max_height="400px"),
        )
        target_activity_selection.observe(on_target_activity_clicked, "value")
        vbox_target_activity_selection = VBox(
            children=[html_descr_target_activity, target_activity_selection]
        )

        html_caption_str = (
            f'<span style="font-weight:'
            f"{self.get_html_str_caption_bold()}; font-size"
            f':{self.caption_size}px">Pick transition activities for '
            f"analysis "
            f"({self.optional_or_required_str})</span>"
        )
        caption_HTML = HTML(html_caption_str)
        hbox_activity_selection = HBox(
            children=[vbox_source_activity_selection, vbox_target_activity_selection]
        )
        box_config = VBox(children=[caption_HTML, hbox_activity_selection])
        self.config_box = box_config
