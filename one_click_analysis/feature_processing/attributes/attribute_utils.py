from typing import List
from typing import Type

from one_click_analysis.feature_processing.attributes.attribute import Attribute
from one_click_analysis.feature_processing.attributes.static_attributes import (
    CaseTableColumnAttribute,
)


def get_attribute_types(attributes: List[Attribute]) -> List[Type[Attribute]]:
    """Get types of attributes (non duplicate)"""
    attribute_types = []
    for attr in attributes:
        if type(attr) not in attribute_types:
            attribute_types.append(type(attr))
    return attribute_types


def remove_duplicates(attributes: List[Attribute]) -> List[Attribute]:
    """Remove duplicate attributes. A duplicated is when two attributes are of the
    same class. So, always the first attribute is kept. If the attribute is an
    instance of CaseTableColumnAttribute, it is also distinguished between table
    names."""
    used_case_table_names = []
    used_attribute_types = []
    non_duplicates = []
    for attr in attributes:
        if (
            isinstance(attr, CaseTableColumnAttribute)
            and attr.table_name not in used_case_table_names
        ):
            used_case_table_names.append(attr.table_name)
            non_duplicates.append(attr)
        elif type(attr) not in used_attribute_types:
            used_attribute_types.append(type(attr))
            non_duplicates.append(attr)
    return non_duplicates
