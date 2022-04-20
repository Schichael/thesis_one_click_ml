from dataclasses import dataclass

from one_click_analysis.feature_processing.attributes_new.attribute import (
    AttributeDataType, Attribute, )


@dataclass
class Feature:
    column_name: str
    datatype: AttributeDataType
    attribute: Attribute

