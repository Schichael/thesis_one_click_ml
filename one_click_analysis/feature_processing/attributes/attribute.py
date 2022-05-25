import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Optional

from pycelonis.celonis_api.pql import pql

from one_click_analysis.process_config.process_config import ProcessConfig


class AttributeType(Enum):
    """Enum to check whether the attribute is a categorical or numerical activity or
    case attribute or none of the two"""

    ACTIVITY_COL = "ACTIVITY_COLL"
    CASE_COL = "CASE_COL"
    OTHER = "OTHER"


class AttributeDataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"

    def __lt__(self, other):
        return self.value <= other.value


class Attribute(abc.ABC):
    """Abstract Attribute class"""

    display_name = "Attribute"

    def __init__(
        self,
        process_config: ProcessConfig,
        attribute_name: str,
        pql_query: pql.PQLColumn,
        data_type: AttributeDataType,
        attribute_type: AttributeType,
        is_feature: bool = False,
        is_class_feature: bool = False,
        unit: str = "",
        column_name: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.process_config = process_config
        self.attribute_name = attribute_name
        self.data_type = data_type
        self.attribute_type = attribute_type
        self.is_feature = is_feature
        self.is_class_feature = is_class_feature
        self.unit = unit
        self.column_name = column_name
        if display_name is not None:
            self.display_name = display_name
        if description is not None:
            self.description = description

    def validate_feature_type(self):
        if self.is_feature and self.is_class_feature:
            raise ValueError(
                "Attributes is_feature and is_class_feature cannot both " "be True"
            )

    @property
    def pql_query(self):
        return self._gen_query()

    @abc.abstractmethod
    def _gen_query(self):
        pass


@dataclass
class AttributeDescriptor:
    """Class with a basic description of an attribute. This is mainly used for
    Processor and the configuration of an analysis."""

    attribute_type: Any
    display_name: str
    description: str
