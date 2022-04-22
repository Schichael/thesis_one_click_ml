import abc
from enum import Enum

from prediction_builder.data_extraction import ProcessModel
from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing.attributes import AttributeDataType


class Attribute(abc.ABC):
    """Abstract Attribute class"""

    def __init__(
        self,
        process_model: ProcessModel,
        attribute_name: str,
        pql_query: pql.PQLColumn,
        data_type: AttributeDataType,
        is_feature: bool = False,
        is_class_feature: bool = False,
        unit: str = "",
    ):
        self.process_model = process_model
        self.attribute_name = attribute_name
        self.pql_query = pql_query
        self.data_type = data_type
        self.is_feature = is_feature
        self.is_class_feature = is_class_feature
        self.unit = unit

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
