import abc
import functools
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from ipywidgets import Box
from ipywidgets import Checkbox
from ipywidgets import DatePicker
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import Select
from ipywidgets import VBox
from ipywidgets import widgets
from pycelonis import get_celonis
from pycelonis.celonis_api.errors import PyCelonisNotFoundError
from pycelonis.celonis_api.pql.pql import PQLFilter

from one_click_analysis import utils
from one_click_analysis.configuration.configurator_class import Configurator
from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDescriptor,
)
from one_click_analysis.feature_processing.feature_processor import (
    FeatureProcessor,
)
from one_click_analysis.process_config.process_config import ProcessConfig


class Configuration(abc.ABC):
    """Abstract class for a configuration"""

    def __init__(
        self,
        configurator: Configurator,
        config_identifier: str,
        additional_prerequsit_config_ids: Optional[List[str]],
        title: str,
        required: bool = False,
        caption_size: int = 14,
        caption_bold: bool = True,
    ):
        self.configurator = configurator

        self.config_identifier = config_identifier
        # Additional config identifiers that need to have an entry before the real
        # config box is built.
        self.additional_prerequsit_config_identifiers = additional_prerequsit_config_ids
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

        # Apply config in case there is a default configuration  # self.apply()

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

    def _validate_additional_prerequisites(self):
        if self.additional_prerequsit_config_identifiers is None:
            return True
        additional_configs_satisfied = True
        for id in self.additional_prerequsit_config_identifiers:
            if id not in self.configurator.config_dict:
                additional_configs_satisfied = False
                break
        return additional_configs_satisfied

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
    """Configuration for defining a start and end date. Start date and end date are
    stored in config['date_start'] and config['date_end]"""

    def __init__(
        self,
        configurator: Configurator,
        datamodel_identifier: str,
        activity_table_identifier: str,
        config_identifier: str = "datepicker",
        additional_prerequsit_config_ids: Optional[List[str]] = None,
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
            additional_prerequsit_config_ids=additional_prerequsit_config_ids,
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
        if self.configurator.config_dict.get("datepicker") is not None:
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

        return (
            process_config_set
            and activity_table_set
            and self._validate_additional_prerequisites()
        )

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
        additional_prerequsit_config_ids: Optional[List[str]] = None,
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
            additional_prerequsit_config_ids=additional_prerequsit_config_ids,
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

        # Create Apply button
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
        if not self.required:
            return True
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
        return self._validate_additional_prerequisites()


class ActivityTableConfig(Configuration):
    """Configuration for selecting the activity table. The name of the activity
    table is stored in config['activity_table_str']
    """

    def __init__(
        self,
        configurator: Configurator,
        datamodel_identifier: str,
        config_identifier: str = "activity_table",
        additional_prerequsit_config_ids: Optional[List[str]] = None,
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
            additional_prerequsit_config_ids=additional_prerequsit_config_ids,
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
        if not self.required:
            return True
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
        return process_config_set and self._validate_additional_prerequisites()


class AttributeSelectionConfig(Configuration):
    """Configuration for selecting the activity table. The static attributes
    descriptors are stored in config['static_attributes']. The dynamic attributes
    descriptors are stored in config['dynamic_attributes'].The selected activity
    table columns are stored in config['activity_table_cols]. The case level table
    columns are stored in config['case_level_table_cols']
    """

    def __init__(
        self,
        configurator: Configurator,
        static_attribute_descriptors: List[AttributeDescriptor],
        dynamic_attribute_descriptors: List[AttributeDescriptor],
        datamodel_identifier: str,
        activity_table_identifier: str,
        config_identifier: str = "attribute_selection",
        additional_prerequsit_config_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        if "title" in kwargs:
            title = kwargs["title"]
            kwargs.pop("title")
        else:
            title = f"Select attributes."

        super().__init__(
            configurator=configurator,
            config_identifier=config_identifier,
            additional_prerequsit_config_ids=additional_prerequsit_config_ids,
            title=title,
            **kwargs,
        )

        self.datamodel_identifier = datamodel_identifier
        self.activity_table_identifier = activity_table_identifier
        self.static_attribute_descriptors = static_attribute_descriptors
        self.dynamic_attribute_descriptors = dynamic_attribute_descriptors
        self.local_selected_static_attributes = self.static_attribute_descriptors.copy()
        self.local_selected_dynamic_attributes = (
            self.dynamic_attribute_descriptors.copy()
        )
        self.local_selected_activity_cols = []  # activity_table_cols.copy()
        self.local_selected_case_cols = {}
        # Initialize config box
        self._create_config_box()

    def _create_initial_local_selected_case_cols(self):
        process_config = self.configurator.config_dict[self.datamodel_identifier][
            "process_config"
        ]
        activity_table_str = self.configurator.config_dict[
            self.activity_table_identifier
        ]["activity_table_str"]
        tables = [
            t.table_str
            for t in process_config.get_case_level_tables(activity_table_str)
        ]
        table_dict = {t: [] for t in tables}
        return table_dict

    def _update_configurator(self):
        self.configurator.config_dict[self.config_identifier] = {
            "static_attributes": self.local_selected_static_attributes,
            "dynamic_attributes": self.local_selected_dynamic_attributes,
            "activity_table_cols": self.local_selected_activity_cols,
            "case_level_table_cols": self.local_selected_case_cols,
        }

    def _create_true_config_box(self):
        """Create the box which contains checkboxes for all attributes including the
        column names of activity and case table (if activity and case table attribute is
        within fp.minor_attrs)

        :return: box with the attribute selection
        """
        process_config = self.configurator.config_dict[self.datamodel_identifier][
            "process_config"
        ]
        activity_table_str = self.configurator.config_dict[
            self.activity_table_identifier
        ]["activity_table_str"]
        self.local_selected_case_cols = self._create_initial_local_selected_case_cols()
        # dict that maps Attribute display_name to the attribute object
        attrs_dict = {
            i.display_name: i
            for i in self.static_attribute_descriptors
            + self.dynamic_attribute_descriptors
        }

        cbs_static = []
        cbs_dynamic = []

        def on_checkbox_clicked(b, selected_attributes):
            """Define behaviour when checkbox of a "normal" attribute (not activity
            or case column attribute) is toggled

            :param b:
            :param selected_attributes: the list of attributes (static or dynamic)
            :return:
            """
            if b.new is False:
                selected_attributes.remove(attrs_dict[b.owner.description])
            else:
                selected_attributes.append(attrs_dict[b.owner.description])
            self._update_configurator()

        on_checkbox_clicked_static = functools.partial(
            on_checkbox_clicked,
            selected_attributes=self.local_selected_static_attributes,
        )
        on_checkbox_clicked_dynamic = functools.partial(
            on_checkbox_clicked,
            selected_attributes=self.local_selected_dynamic_attributes,
        )

        for attr in self.static_attribute_descriptors:
            # if the attribute is the label,
            cb = widgets.Checkbox(
                value=True, description=attr.display_name, indent=False
            )
            cb.observe(on_checkbox_clicked_static, "value")
            cbs_static.append(cb)
        for attr in self.dynamic_attribute_descriptors:
            # if the attribute is the label,
            cb = widgets.Checkbox(
                value=True, description=attr.display_name, indent=False
            )
            cb.observe(on_checkbox_clicked_dynamic, "value")
            cbs_dynamic.append(cb)
        # activity_table_cols_cat, activity_table_cols_num = \
        #    process_config.get_categorical_numerical_columns(
        #    activity_table_str)

        # activity_table_cols = activity_table_cols_cat + activity_table_cols_num
        # activity_table_cols = sorted(activity_table_cols)
        activity_table = process_config.table_dict[activity_table_str]
        cbs_activity_table = self.create_cbs_activity_case(
            "activity", table=activity_table
        )
        cbs_case_tables = []
        case_level_tables = process_config.get_case_level_tables(activity_table_str)
        for table in case_level_tables:
            cbs_case_table = self.create_cbs_activity_case("case", table=table)
            cbs_case_tables.append(cbs_case_table)

        cbs_activity_case_table = [cbs_activity_table] + cbs_case_tables
        # remove None
        cbs_activity_case_table = list(filter(None, cbs_activity_case_table))

        # Update configurator with default values
        self._update_configurator()
        static_header_str = (
            '<span style="font-weight:bold;  font-size:14px">Static attributes '
            "</span>"
        )
        static_header = widgets.HTML(static_header_str)

        dynamic_header_str = (
            '<span style="font-weight:bold;  font-size:14px">Dynamic attributes '
            "</span>"
        )
        dynamic_header = widgets.HTML(dynamic_header_str)

        vbox_cbs_static = widgets.VBox(children=[static_header] + cbs_static)
        vbox_cbs_dynamic = widgets.VBox(children=[dynamic_header] + cbs_dynamic)

        vbox_cbs = widgets.VBox(
            children=[vbox_cbs_static, vbox_cbs_dynamic] + cbs_activity_case_table
        )

        html_title = widgets.HTML(self.html_caption_str)
        vbox_all_attributes = widgets.VBox(children=[html_title, vbox_cbs])

        self.config_box = vbox_all_attributes

    def create_cbs_activity_case(self, table_type: str, table: Any) -> widgets.VBox:
        """Create the checkboxes for activity or case table columns.

        :param table_type: type of the table to use. Either 'case' for the case table or
        'activity' for the activity table
        :return: box with the column attribute checkboxes
        """
        if table_type == "case" and table is None:
            raise ValueError("When table_type is 'case', table_name must not be None")

        process_config = self.configurator.config_dict[self.datamodel_identifier][
            "process_config"
        ]
        table_name = table.table_str

        cat_columns, num_columns = process_config.get_categorical_numerical_columns(
            table_name
        )

        columns = cat_columns + num_columns

        column_dict = {col.name + " (categorical)": col.name for col in cat_columns}
        column_dict_numerical = {
            col.name + " (numeric)": col.name for col in num_columns
        }
        column_dict.update(column_dict_numerical)
        # Configs based on table value
        if table_type == "activity":
            title = "Selection of Activity table columns to be usable by attributes:"
            selected_cols = self.local_selected_activity_cols
        elif table_type == "case":
            title = (
                f"Selection of {table_name} table columns to be usable by "
                f"attributes:"
            )
            selected_cols = self.local_selected_case_cols[table_name]
        else:
            raise ValueError("table must be one of ['case', 'activity']")

        # Checkboxes for columns
        cbs = []

        # Add checkbox to Select/Unselect all columms as first checkbox
        cb_select_all = widgets.Checkbox(
            value=True, description="Select / Unselect all", indent=False
        )

        cbs.append(cb_select_all)

        # Define behaviour that happens when the Select / Unselect all checkbox is
        # toggled
        def select_all_changed(b, cbs: List[widgets.Checkbox]):
            """Define behaviour that happens when the Select / Unselect all checkbox
            is toggled.
            If the 'select/unselect all' checkbox is toggled, all other checkboxes are
            set to the same value

            :param b: needed for observing
            :param cbs: list with the column checkboxes
            :return:
            """
            for cb in cbs:
                cb.value = b.new

        select_all_changed = functools.partial(select_all_changed, cbs=cbs)
        cb_select_all.observe(select_all_changed, "value")

        # Define behaviour that happens when a column checkbox is toggled
        def column_cb_changed(
            b,
            cb_select_all: widgets.Checkbox,
            fct_select_all: Callable,
            selected_cols: List[str],
            all_cols: List[str],
        ):
            """Define behaviour that happens when a column checkbox is toggled.
            If a checkbox is unselected, the corresponding column is removed from the
            list of selected columns.
            Also, the Select/Unselect all checkbox will be unselected.
            If a checkbox is selected, the corresponding column is appended to the
            list of selected columns.
            Also, the Select/Unselect all checkbox will be selected if this event
            leads to all checkboxes selected.

            :param b: needed for observing
            :param cb_select_all: the Select/Unselect all checkbox
            :param fct_select_all: function that is observed by the Select/Unselect
            all checkbox
            :param selected_cols: list with the selected columns
            :param all_cols: list with all columns
            :return:
            """

            # Unobserve the Select/Unselect all checkbox as an unselect of a column
            # checkbox would lead to all checkboxes being unselected else
            cb_select_all.unobserve(fct_select_all, "value")
            if b.new is False:
                selected_cols.remove(column_dict[b.owner.description])
                cb_select_all.value = False
            else:
                selected_cols.append(column_dict[b.owner.description])
                if len(selected_cols) == len(all_cols):
                    cb_select_all.value = True
            # Observe the Select/Unselect all checkbox
            cb_select_all.observe(fct_select_all, "value")
            self._update_configurator()

        column_cb_changed = functools.partial(
            column_cb_changed,
            cb_select_all=cb_select_all,
            fct_select_all=select_all_changed,
            selected_cols=selected_cols,
            all_cols=columns,
        )

        # Construct column checkboxes
        for col_key in sorted(list(column_dict.keys())):
            cb = widgets.Checkbox(value=False, description=col_key, indent=False)
            cb.observe(column_cb_changed, "value")
            cbs.append(cb)

        # Create VBoxes with the checkboxes
        layout_cb_box = widgets.Layout(max_height="300px")
        vbox_cbs_cat = widgets.VBox(children=cbs, layout=layout_cb_box)

        # Create accordion
        acc = widgets.Accordion(children=[vbox_cbs_cat], selected_index=None)
        acc.set_title(0, "columns")

        # Add title to accordion using a VBox
        html_title_str = (
            '<span style="font-weight:bold;  font-size:14px">' + title + "</span>"
        )
        html_title = widgets.HTML(html_title_str)

        vbox_col_attrs = widgets.VBox(children=[html_title, acc])
        return vbox_col_attrs

    @property
    def requirement_met(self) -> bool:
        if not self.required:
            return True
        try:
            if self.configurator.config_dict[self.config_identifier] is not None:
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
        activity_table_set = (
            self.activity_table_identifier in self.configurator.config_dict
            and self.configurator.config_dict[self.activity_table_identifier]
            is not None
        )
        return (
            process_config_set
            and activity_table_set
            and self._validate_additional_prerequisites()
        )


class TransitionConfig(Configuration):
    """Configuration for defining a source activity and a target activity. The name
    of the source activity is stored in config['source_activity']. The name
    of the target activity is stored in config['target_activity']
    """

    def __init__(
        self,
        configurator: Configurator,
        datamodel_identifier: str,
        activitytable_identifier: str,
        config_identifier: str = "transition",
        additional_prerequsit_config_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        if "title" in kwargs:
            title = kwargs["title"]
            kwargs.pop("title")
        else:
            title = f"Pick transition activities for analysis"

        super().__init__(
            configurator=configurator,
            config_identifier=config_identifier,
            additional_prerequsit_config_ids=additional_prerequsit_config_ids,
            title=title,
            **kwargs,
        )

        self.datamodel_identifier = datamodel_identifier
        self.activitytable_identifier = activitytable_identifier

        # Initialize config box
        self._create_config_box()

    def _create_true_config_box(self):
        process_config = self.configurator.config_dict[self.datamodel_identifier][
            "process_config"
        ]
        activity_table_str = self.configurator.config_dict[
            self.activitytable_identifier
        ]["activity_table_str"]
        # Create ipywidgets Box object for configuration visualization
        html_descr_source_activity = HTML(
            '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px; '
            'font-size: 14px;">Pick a source activity</div>'
        )
        html_descr_target_activity = HTML(
            '<div style="line-height:140%; margin-top: 0px; margin-bottom: 0px; '
            'font-size: 14px;">Pick a target activity</div>'
        )
        activity_table = process_config.table_dict[activity_table_str]
        activities = activity_table.process_model.activities
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

        # Create Apply button
        apply_button = widgets.Button(description="Apply")

        def on_apply_clicked(b):
            if (
                self.config.get("source_activity") is not None
                and self.config.get("target_activity") is not None
            ):
                self.apply()

        apply_button.on_click(on_apply_clicked)

        # html_caption_str = (f'<span style="font-weight:'
        #                    f"{self.get_html_str_caption_bold()}; font-size"
        #                    f':{self.caption_size}px">Pick transition activities for '
        #                    f"analysis "
        #                    f"({self.optional_or_required_str})</span>")
        caption_HTML = HTML(self.html_caption_str)
        hbox_activity_selection = HBox(
            children=[vbox_source_activity_selection, vbox_target_activity_selection]
        )
        box_config = VBox(
            children=[caption_HTML, hbox_activity_selection, apply_button]
        )
        self.config_box = box_config

    @property
    def requirement_met(self) -> bool:
        if not self.required:
            return True
        if self.configurator.config_dict.get(self.config_identifier) is not None:
            return True
        else:
            return False

    def validate_prerequisites(self) -> bool:
        process_config_set = (
            self.configurator.config_dict.get(self.datamodel_identifier) is not None
        )
        activity_table_set = (
            self.configurator.config_dict.get(self.activitytable_identifier) is not None
        )
        return (
            process_config_set
            and activity_table_set
            and self._validate_additional_prerequisites()
        )


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
