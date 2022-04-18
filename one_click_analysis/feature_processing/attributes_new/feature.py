from dataclasses import dataclass

from one_click_analysis.feature_processing.attributes.attribute import AttributeDataType


@dataclass
class Feature:
    column_name: str
    datatype: AttributeDataType
