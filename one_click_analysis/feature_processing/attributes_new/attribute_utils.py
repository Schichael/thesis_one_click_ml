from typing import List

from one_click_analysis.feature_processing.attributes_new.attribute import Attribute


def get_aggregation_df_name(agg: str):
    """Generate the name of the aggregation to display to the user from the
    aggregation String that is used for a PQL query

    :param agg: original aggregation name as used for
    :return: aggregation string to display
    """
    if agg == "MIN":
        return "minimum"
    elif agg == "MAX":
        return "maximum"
    elif agg == "AVG":
        return "mean"
    elif agg == "MEDIAN":
        return "median"
    elif agg == "FIRST":
        return "first"
    elif agg == "LAST":
        return "last"


def remove_duplicate_attributes(attributes: List[Attribute]) -> List[Attribute]:
    """Remove duplicate attributes. A duplicated is when two attributes are of the
    same class."""
    used_attribute_types = []
    non_duplicates = []
    for attr in attributes:
        if type(attr) not in used_attribute_types:
            used_attribute_types.append(type(attr))
            non_duplicates.append(attr)
    return non_duplicates
