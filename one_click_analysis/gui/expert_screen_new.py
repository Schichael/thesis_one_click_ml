import functools
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import ipywidgets as widgets

from one_click_analysis.attribute_selection import AttributeSelection
from one_click_analysis.feature_processing.attributes.attribute import (
    Attribute,
)
from one_click_analysis.feature_processing.attributes.attribute_utils import (
    remove_duplicates,
)
from one_click_analysis.feature_processing.attributes.feature import Feature


class ExpertScreen:
    """Contains attribute selection"""

    def __init__(
        self,
        attributes: List[Attribute],
        activity_table_cols: List[str],
        case_table_cols: Dict[str, List[str]],
        # categorical_case_table_cols: List[str],
        # numerical_case_table_cols: List[str],
        features: List[Feature],
        attr_selection: AttributeSelection,
    ):
        """
        :param fp: FeatureProcessor with processed features
        :param attr_selection: AttributeSelection object

        """
        self.attributes = remove_duplicates(attributes)
        # case_table_cols has the form
        # "tablename":
        #   "columns":
        #       ["colname_1", "colname_2"]
        self.activity_table_cols = activity_table_cols
        self.case_table_cols = case_table_cols
        # self.categorical_activity_table_cols = categorical_activity_table_cols
        # self.numerical_activity_table_cols = numerical_activity_table_cols
        # self.categorical_case_table_cols = categorical_case_table_cols
        # self.numerical_case_table_cols = numerical_case_table_cols
        self.features = features
        self.attr_selection = attr_selection
        self.expert_box = None

        # variables to store the local selected attributes and columns. Right now
        # take all from the FeatureProcessor
        self.local_selected_attributes = self.attributes.copy()

        # self.local_selected_activity_cat_cols = categorical_activity_table_cols.copy()
        # self.local_selected_activity_num_cols = numerical_activity_table_cols.copy()
        self.local_selected_activity_cols = activity_table_cols.copy()
        self.local_selected_case_cols = case_table_cols.copy()
        # self.local_selected_case_cat_cols = categorical_case_table_cols.copy()
        # self.local_selected_case_num_cols = numerical_case_table_cols.copy()

    def create_expert_box(self):
        """Create the box which contains checkboxes for all attributes including the
        column names of activity and case table (if activity and case table attribute is
        within fp.minor_attrs)

        :return: box with the attribute selection
        """
        # dict that maps Attribute display_name to the attribute object
        attrs_dict = {i.display_name: i for i in self.attributes}

        cbs = []

        def on_checkbox_clicked(b):
            """Define behaviour when checkbox of a "normal" attribute (not activity
            or case column attribute) is toggled

            :param b:
            :return:
            """
            if b.new is False:
                self.local_selected_attributes.remove(attrs_dict[b.owner.description])
            else:
                self.local_selected_attributes.append(attrs_dict[b.owner.description])

        for attr in self.attributes:
            # if the attribute is the label,
            if not attr.is_feature:
                continue
            cb = widgets.Checkbox(
                value=True, description=attr.display_name, indent=False
            )
            cb.observe(on_checkbox_clicked, "value")
            cbs.append(cb)

        cbs_activity_table = self.create_cbs_activity_case(
            "activity", columns=self.activity_table_cols
        )
        cbs_case_tables = []
        for table_name, cols in self.case_table_cols.items():
            cbs_case_table = self.create_cbs_activity_case(
                "case", columns=cols, table_name=table_name
            )
            cbs_case_tables.append(cbs_case_table)

        cbs_activity_case_table = [cbs_activity_table] + cbs_case_tables
        # remove None
        cbs_activity_case_table = list(filter(None, cbs_activity_case_table))

        vbox_cbs = widgets.VBox(children=cbs + cbs_activity_case_table)
        button_apply = widgets.Button(description="Apply Changes")

        def on_button_apply_clicked(b):
            """When the button button_apply is clicked, the variables
            self.selected_attributes, self.selected_activity_table_cols and
            self.selected_case_table_cols are overwritten by the currently selected
            attributes and columns

            :param b: needed for observing
            :return:
            """
            self.attr_selection.selected_attributes = self.local_selected_attributes
            self.attr_selection.selected_activity_table_cols = (
                self.local_selected_activity_cat_cols
                + self.local_selected_activity_num_cols
            )
            self.attr_selection.selected_case_table_cols = (
                self.local_selected_case_cat_cols + self.local_selected_case_num_cols
            )
            self.attr_selection.update()

        button_apply.on_click(on_button_apply_clicked)

        # Add a title
        html_title_str = (
            '<span style="font-weight:bold;  font-size:16px">Attribute '
            "selection</span>"
        )
        html_title = widgets.HTML(html_title_str)
        vbox_all_attributes = widgets.VBox(
            children=[html_title, vbox_cbs, button_apply]
        )
        self.expert_box = vbox_all_attributes

    def create_cbs_activity_case(
        self, table_type: str, columns: List[str], table_name: Optional[str] = None
    ) -> widgets.VBox:
        """Create the checkboxes for activity or case table columns.

        :param table_type: type of the table to use. Either 'case' for the case table or
        'activity' for the activity table
        :param columns: list with the columns
        :param table_name: table_name. Needed when it's not the activity table
        :return: box with the column attribute checkboxes
        """
        if table_type == "case" and table_name is None:
            raise ValueError("When table_type is 'case', table_name must not be None")

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
            If the select/unselect all checkbox is toggled, all other checkboxes are
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
                selected_cols.remove(b.owner.description)
                cb_select_all.value = False
            else:
                selected_cols.append(b.owner.description)
                if len(selected_cols) == len(all_cols):
                    cb_select_all.value = True
            # Observe the Select/Unselect all checkbox
            cb_select_all.observe(fct_select_all, "value")

        column_cb_changed = functools.partial(
            column_cb_changed,
            cb_select_all=cb_select_all,
            fct_select_all=select_all_changed,
            selected_cols=selected_cols,
            all_cols=columns,
        )

        # Construct column checkboxes
        for col in sorted(columns):
            cb = widgets.Checkbox(value=True, description=col, indent=False)
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
