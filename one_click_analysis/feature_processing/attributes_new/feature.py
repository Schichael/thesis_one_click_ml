from dataclasses import dataclass
from dataclasses import field
from typing import Optional

from one_click_analysis.feature_processing.attributes_new.attribute import Attribute
from one_click_analysis.feature_processing.attributes_new.attribute import (
    AttributeDataType,
)


@dataclass
class Feature:
    column_name: str
    datatype: AttributeDataType
    attribute: Attribute
    # If the feature was generated from a categorical attribute using ohe,
    # an attribute_value can be given to be used later.
    attribute_value: Optional[str] = None
    metrics: dict = field(default_factory=lambda: {})
