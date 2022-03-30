import functools
from typing import Callable
from typing import List

import ipywidgets as widgets

from one_click_analysis.attribute_selection import AttributeSelection
from one_click_analysis.feature_processing import attributes
from one_click_analysis.feature_processing.feature_processor import FeatureProcessor


class ExpertScreen:
    """Contains attribute selection"""

    def __init__(
        self,
        fp: FeatureProcessor,
        attr_selection: AttributeSelection,
    ):
        """
        :param fp: FeatureProcessor with processed features
        :param attr_selection: AttributeSelection object

        """
        self.fp = fp
        self.attr_selection = attr_selection
        self.expert_box = None

        # variables to store the local selected attributes and columns. Right now
        # take all from the FeatureProcessor
        self.local_selected_attributes = fp.minor_attrs.copy()

        self.local_selected_activity_cat_cols = fp.dynamic_categorical_cols.copy()
        self.local_selected_activity_num_cols = fp.dynamic_numerical_cols.copy()

        self.local_selected_case_cat_cols = fp.static_categorical_cols.copy()
        self.local_selected_case_num_cols = fp.static_numerical_cols.copy()

    def create_expert_box(self):
        """Create the box which contains checkboxes for all attributes including the
        column names of activity and case table (if activity and case table attribute is
        within fp.minor_attrs)

        :return: box with the attribute selection
        """
        # dict that maps MinorAttribute attribute_name to the attribute object
        minor_attrs_dict = {i.attribute_name: i for i in self.fp.minor_attrs}

        cbs = []
        cbs_activity_table = None
        cbs_case_table = None

        def on_checkbox_clicked(b):
            """Define behaviour when checkbox of a "normal" attribute (not activity
            or case column attribute) is toggled

            :param b:
            :return:
            """
            if b.new is False:
                self.local_selected_attributes.remove(
                    minor_attrs_dict[b.owner.description]
                )
            else:
                self.local_selected_attributes.append(
                    minor_attrs_dict[b.owner.description]
                )

        for attr in self.fp.minor_attrs:
            # if the attribute is the label,
            if not isinstance(
                attr, attributes.ActivityTableColumnMinorAttribute
            ) and not isinstance(attr, attributes.CaseTableColumnMinorAttribute):
                cb = widgets.Checkbox(
                    value=True, description=attr.attribute_name, indent=False
                )
                cb.observe(on_checkbox_clicked, "value")
                cbs.append(cb)
            elif isinstance(attr, attributes.ActivityTableColumnMinorAttribute):
                (cbs_activity_table) = self.create_cbs_activity_case(
                    "activity",
                    cat_columns=self.fp.dynamic_categorical_cols,
                    num_columns=self.fp.dynamic_numerical_cols,
                )
            elif isinstance(attr, attributes.CaseTableColumnMinorAttribute):
                cbs_case_table = self.create_cbs_activity_case(
                    "case",
                    cat_columns=self.fp.static_categorical_cols,
                    num_columns=self.fp.static_numerical_cols,
                )

        cbs_activity_case_table = [cbs_activity_table, cbs_case_table]
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
        self, table: str, cat_columns: List[str], num_columns: List[str]
    ) -> widgets.VBox:
        """Create the checkboxes for activity or case table columns.

        :param table: type of the table to use. Either 'case' for the case table or
        'activity' for the activity table
        :param cat_columns: list with the categorical columns
        :param num_columns: list with the numerical columns
        :return: box with the column attribute checkboxes
        """
        # Configs based on table value
        if table == "activity":
            title = "Activity table attributes"
            selected_cat_cols = self.local_selected_activity_cat_cols
            selected_num_cols = self.local_selected_activity_num_cols
        elif table == "case":
            title = "Case table attributes"
            selected_cat_cols = self.local_selected_case_cat_cols
            selected_num_cols = self.local_selected_case_num_cols
        else:
            raise ValueError("table must be one of ['case', 'activity']")

        # Checkboxes for categorical and numerical columns
        cbs_cat = []
        cbs_num = []

        # Add checkbox to Select/Unselect all columms as first checkbox
        cb_select_all_cat = widgets.Checkbox(
            value=True, description="Select / Unselect all", indent=False
        )
        cb_select_all_num = widgets.Checkbox(
            value=True, description="Select / Unselect all", indent=False
        )
        cbs_cat.append(cb_select_all_cat)
        cbs_num.append(cb_select_all_num)

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

        select_all_changed_cat = functools.partial(select_all_changed, cbs=cbs_cat)
        cb_select_all_cat.observe(select_all_changed_cat, "value")
        select_all_changed_num = functools.partial(select_all_changed, cbs=cbs_num)
        cb_select_all_num.observe(select_all_changed_num, "value")

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

        column_cb_changed_cat = functools.partial(
            column_cb_changed,
            cb_select_all=cb_select_all_cat,
            fct_select_all=select_all_changed_cat,
            selected_cols=selected_cat_cols,
            all_cols=cat_columns,
        )

        column_cb_changed_num = functools.partial(
            column_cb_changed,
            cb_select_all=cb_select_all_num,
            fct_select_all=select_all_changed_num,
            selected_cols=selected_num_cols,
            all_cols=num_columns,
        )

        # Construct column checkboxes for categorical and numerical columns
        for col in sorted(cat_columns):
            cb = widgets.Checkbox(value=True, description=col, indent=False)
            cb.observe(column_cb_changed_cat, "value")
            cbs_cat.append(cb)

        for col in sorted(num_columns):
            cb = widgets.Checkbox(value=True, description=col, indent=False)
            cb.observe(column_cb_changed_num, "value")
            cbs_num.append(cb)

        # Create VBoxes with the checkboxes
        layout_cb_box = widgets.Layout(max_height="300px")
        vbox_cbs_cat = widgets.VBox(children=cbs_cat, layout=layout_cb_box)
        vbox_cbs_num = widgets.VBox(children=cbs_num, layout=layout_cb_box)

        # Create accordion
        acc = widgets.Accordion(
            children=[vbox_cbs_cat, vbox_cbs_num], selected_index=None
        )
        acc.set_title(0, "categorical")
        acc.set_title(1, "numerical")

        # Add title to accordion using a VBox
        html_title_str = (
            '<span style="font-weight:bold;  font-size:14px">' + title + "</span>"
        )
        html_title = widgets.HTML(html_title_str)

        vbox_col_attrs = widgets.VBox(children=[html_title, acc])
        return vbox_col_attrs
