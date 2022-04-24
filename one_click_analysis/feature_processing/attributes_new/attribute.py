import abc
from enum import Enum
from typing import Optional

from prediction_builder.data_extraction import ProcessModel
from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing.attributes import AttributeDataType


class AttributeType(Enum):
    """Enum to check whether the attribute is a categorical or numerical activity or
    case attribute or none of the two"""

    ACTIVITY_COL_NUMERICAL = "ACTIVITY_NUMERICAL"
    ACTIVITY_COL_CATEGORICAL = "ACTIVITY_COL_CATEGORICAL"
    CASE_COL_NUMERICAL = "CASE_COL_NUMERICAL"
    CASE_COL_CATEGORICAL = "CASE_COL_CATEGORICAL"
    OTHER = "OTHER"


class Attribute(abc.ABC):
    """Abstract Attribute class"""

    display_name = "Attribute"

    def __init__(
        self,
        process_model: ProcessModel,
        attribute_name: str,
        pql_query: pql.PQLColumn,
        data_type: AttributeDataType,
        attribute_type: AttributeType,
        is_feature: bool = False,
        is_class_feature: bool = False,
        unit: str = "",
        column_name: Optional[str] = None,
    ):
        self.process_model = process_model
        self.attribute_name = attribute_name
        self.pql_query = pql_query
        self.data_type = data_type
        self.attribute_type = attribute_type
        self.is_feature = is_feature
        self.is_class_feature = is_class_feature
        self.unit = unit
        self.column_name = column_name

    def validate_feature_type(self):
        if self.is_feature and self.is_class_feature:
            raise ValueError(
                "Attributes is_feature and is_class_feature cannot both " "be True"
            )

    @abc.abstractmethod
    def _gen_query(self):
        pass


class AttributeDataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"

    def __lt__(self, other):
        return self.value <= other.value
