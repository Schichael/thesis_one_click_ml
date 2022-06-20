import functools
from collections import OrderedDict
from typing import Callable
from typing import List

import ipywidgets as widgets
from ipywidgets import HTML
from ipywidgets import VBox

from one_click_analysis.feature_processing.attributes.attribute import Attribute
from one_click_analysis.feature_processing.attributes.attribute import AttributeType
from one_click_analysis.feature_processing.attributes.feature import Feature


class FeatureSelection:
    """Contains the feature selection ui component"""

    def __init__(
        self,
        attributes: List[Attribute],
        features: List[Feature],
    ):
        self.attributes = attributes
        self.features = features
        self.selected_features = features.copy()
        self.attribute_feature_dict = self._create_attribute_feature_dict()
        self.selection_box = self._create_selection()

    def _create_attribute_feature_dict(self):
        """Create dictionary that maps attributes to features"""
        attribute_display_names = [attr.display_name for attr in self.attributes]
        attribute_display_names = list(dict.fromkeys(attribute_display_names))

        attribute_feature_dict = OrderedDict()
        for add_display_name in attribute_display_names:
            features = [
                f for f in self.features if f.attribute.display_name == add_display_name
            ]
            attribute_feature_dict[add_display_name] = features
        return attribute_feature_dict

    def _create_column_value_dict(self, features: List[Feature]):
        """Create dict that maps columns to features"""
        column_feature_dict = {}
        columns = {f.attribute.column_name for f in features}
        for col in columns:
            features = [f for f in features if f.attribute.column_name == col]
            column_feature_dict[col] = features
        return column_feature_dict

    def _create_selection(self):
        accordions = []
        for attribute_dname, features in self.attribute_feature_dict.items():
            # check if features' attribute is a activity or case level table attribute
            if len(features) == 0:
                continue
            if features[0].attribute.attribute_type in [
                AttributeType.ACTIVITY_COL,
                AttributeType.CASE_COL,
            ]:
                column_dict = self._create_column_value_dict(features)
                acc = self._create_selection_tables(
                    attr_display_name=attribute_dname, column_dict=column_dict
                )
                accordions = accordions + acc
            else:
                acc = self._create_selection_normal(
                    attr_display_name=attribute_dname, features=features
                )
                accordions.append(acc)

        title = (
            f'<div style="line-height:140%;font-weight:bold; font-size: '
            f'14px">Select features to use'
        )
        title_html = HTML(title)
        acc = widgets.Accordion(children=accordions, selected_index=None)
        acc.set_title(0, "Select features")
        selection_box = VBox(children=[title_html, acc])
        return selection_box

    def _create_selection_normal(self, attr_display_name: str, features):
        """Create selections for normal attribute features"""
        title = f"{attr_display_name}"

        if features[0].attribute_value is not None:
            feature_value_dict = {
                feature.attribute_value: feature for feature in features
            }
        else:
            feature_value_dict = {
                feature.df_column_name: feature for feature in features
            }

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

        def value_cb_changed(
            b,
            cb_select_all: widgets.Checkbox,
            fct_select_all: Callable,
            all_features: List[str],
        ):
            """Define behaviour that happens when a value checkbox is toggled.
            If a checkbox is unselected, the corresponding feature is removed from
            the
            list of selected features.
            Also, the Select/Unselect all checkbox will be unselected.
            If a checkbox is selected, the corresponding feature is appended to the
            list of selected features.
            Also, the Select/Unselect all checkbox will be selected if this event
            leads to all checkboxes selected.

            :param b: needed for observing
            :param cb_select_all: the Select/Unselect all checkbox
            :param fct_select_all: function that is observed by the Select/Unselect
            all checkbox
            :param all_features: list with all columns
            :return:
            """

            # Unobserve the Select/Unselect all checkbox as an unselect of a column
            # checkbox would lead to all checkboxes being unselected else
            cb_select_all.unobserve(fct_select_all, "value")
            if b.new is False:
                self.selected_features.remove(feature_value_dict[b.owner.description])
                cb_select_all.value = False
            else:
                self.selected_features.append(feature_value_dict[b.owner.description])
                if len(self.selected_features) == len(all_features):
                    cb_select_all.value = True
            # Observe the Select/Unselect all checkbox
            cb_select_all.observe(fct_select_all, "value")

        selected_features = features.copy()

        cbs = []
        cb_select_all_value = widgets.Checkbox(
            value=True, description="Select / Unselect all", indent=False
        )
        select_all_changed = functools.partial(select_all_changed, cbs=cbs)
        cb_select_all_value.observe(select_all_changed, "value")

        cbs.append(cb_select_all_value)

        value_cb_changed = functools.partial(
            value_cb_changed,
            cb_select_all=cb_select_all_value,
            fct_select_all=select_all_changed,
            selected_features=selected_features,
            all_features=features,
            feature_value_dict=feature_value_dict,
        )

        sorted_values = sorted(feature_value_dict.keys())
        cbs_vals = []
        for val in sorted_values:
            cb = widgets.Checkbox(value=True, description=val, indent=False)
            cb.observe(value_cb_changed, "value")
            cbs_vals.append(cb)
        cbs = cbs + cbs_vals
        # Create VBoxes with the checkboxes
        layout_cb_box = widgets.Layout(max_height="300px")
        vbox_cbs_cat = widgets.VBox(children=cbs, layout=layout_cb_box)

        acc = widgets.Accordion(children=[vbox_cbs_cat], selected_index=None)
        acc.set_title(0, title)
        return acc

    def _create_selection_tables(self, attr_display_name: str, column_dict: dict):
        """Create selections for table column features"""
        title = f"{attr_display_name}"

        # Checkboxes for columns

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

        def value_cb_changed(
            b,
            cb_select_all: widgets.Checkbox,
            fct_select_all: Callable,
            all_features: List[str],
            feature_value_dict: dict,
        ):
            """Define behaviour that happens when a value checkbox is toggled.
            If a checkbox is unselected, the corresponding feature is removed from
            the
            list of selected features.
            Also, the Select/Unselect all checkbox will be unselected.
            If a checkbox is selected, the corresponding feature is appended to the
            list of selected features.
            Also, the Select/Unselect all checkbox will be selected if this event
            leads to all checkboxes selected.

            :param b: needed for observing
            :param cb_select_all: the Select/Unselect all checkbox
            :param fct_select_all: function that is observed by the Select/Unselect
            all checkbox
            :param all_features: list with all columns
            :return:
            """

            # Unobserve the Select/Unselect all checkbox as an unselect of a column
            # checkbox would lead to all checkboxes being unselected else
            cb_select_all.unobserve(fct_select_all, "value")
            if b.new is False:
                self.selected_features.remove(feature_value_dict[b.owner.description])
                cb_select_all.value = False
            else:
                self.selected_features.append(feature_value_dict[b.owner.description])
                if len(self.selected_features) == len(all_features):
                    cb_select_all.value = True
            # Observe the Select/Unselect all checkbox
            cb_select_all.observe(fct_select_all, "value")

        col_accordions = []
        cbs_all = []
        for col, features in column_dict.items():
            if len(features) == 0:
                continue
            if features[0].attribute_value is not None:
                feature_value_dict = {
                    feature.attribute_value: feature for feature in features
                }
            else:
                feature_value_dict = {
                    feature.df_column_name: feature for feature in features
                }

            selected_features = features.copy()

            cbs = []
            cb_select_all_value = widgets.Checkbox(
                value=True, description="Select / Unselect all", indent=False
            )
            select_all_changed = functools.partial(select_all_changed, cbs=cbs)
            cb_select_all_value.observe(select_all_changed, "value")

            cbs.append(cb_select_all_value)

            value_cb_changed = functools.partial(
                value_cb_changed,
                cb_select_all=cb_select_all_value,
                fct_select_all=select_all_changed,
                selected_features=selected_features,
                all_features=features,
                feature_value_dict=feature_value_dict,
            )

            sorted_values = sorted(feature_value_dict.keys())
            cbs_vals = []
            for val in sorted_values:
                cb = widgets.Checkbox(value=True, description=val, indent=False)
                cb.observe(value_cb_changed, "value")
                cbs_vals.append(cb)
            cbs = cbs + cbs_vals
            # Create VBoxes with the checkboxes
            layout_cb_box = widgets.Layout(max_height="300px")
            vbox_cbs_cat = widgets.VBox(children=cbs, layout=layout_cb_box)

            acc = widgets.Accordion(children=[vbox_cbs_cat], selected_index=None)
            acc.set_title(0, col)
            col_accordions.append(acc)
            cbs_all = cbs_all + cbs_vals

        cb_select_all_columns = widgets.Checkbox(
            value=True, description="Select / Unselect all " "columns", indent=False
        )
        select_all_changed = functools.partial(select_all_changed, cbs=cbs_all)
        cb_select_all_columns.observe(select_all_changed, "value")

        acc = widgets.Accordion(
            children=[cb_select_all_columns] + col_accordions, selected_index=None
        )
        acc.set_title(0, title)
        col_accordions.append(acc)
        return col_accordions
