import abc
from enum import Enum

from prediction_builder.data_extraction import ProcessModel
from pycelonis.celonis_api.pql import pql


class Attribute(abc.ABC):
    """Abstract Attribute class"""

    def __init__(
        self,
        process_model: ProcessModel,
        attribute_name: str,
        pql_query: pql.PQLColumn,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.attribute_name = attribute_name
        self.pql_query = pql_query
        self.is_feature = is_feature
        self.is_class_feature = is_class_feature

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

    def __lt__(self, other):
        return self.value <= other.value
