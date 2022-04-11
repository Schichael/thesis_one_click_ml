import abc
from dataclasses import dataclass
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class MajorAttribute(Enum):
    ACTIVITY = "Activity"
    CASE = "Case"
    ordering = [ACTIVITY, CASE]

    def __lt__(self, other):
        return self.value <= other.value


class MinorAttribute(abc.ABC):
    def __init__(
        self,
        attribute_name: str,
        major_attribute: MajorAttribute,
        is_label: bool = False,
    ):
        self.is_label = is_label
        self.attribute_name = attribute_name
        self.major_attribute = major_attribute


class CaseDurationMinorAttribute(MinorAttribute):
    def __init__(self, time_aggregation: str, is_label: bool = False):
        self.time_aggregation = time_aggregation
        attribute_name = "Case duration"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute, is_label)


class WorkInProgressMinorAttribute(MinorAttribute):
    def __init__(self, aggregations: Union[str, List[str]], is_label: bool = False):
        self.aggregations = aggregations
        attribute_name = "Work in Progress"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute, is_label)


class EventCountMinorAttribute(MinorAttribute):
    def __init__(self, is_label: bool = False):
        attribute_name = "Event count"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute, is_label)


class ReworkOccurenceMinorAttribute(MinorAttribute):
    def __init__(self, is_label: bool = False):
        attribute_name = "Rework"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute, is_label)


class ActivityOccurenceMinorAttribute(MinorAttribute):
    def __init__(self, is_label: bool = False):
        attribute_name = "Activity occurence"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute, is_label)


class ActivityCountMinorAttribute(MinorAttribute):
    def __init__(self, is_label: bool = False):
        attribute_name = "Activity Count"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute, is_label)


class EndActivityMinorAttribute(MinorAttribute):
    def __init__(self, is_label: bool = False):
        attribute_name = "End activity"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute, is_label)


class StartActivityMinorAttribute(MinorAttribute):
    def __init__(self, is_label: bool = False):
        attribute_name = "Start activity"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute, is_label)


class ActivityTableColumnMinorAttribute(MinorAttribute):
    def __init__(self, aggregations: Optional[Union[List[str], str]] = None):
        self.aggregations = aggregations
        attribute_name = "Activity table column"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute)


class CaseTableColumnMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Case table column"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute)


class TransitionMinorAttribute(MinorAttribute):
    def __init__(
        self, transitions: List[Tuple[str, List[str]]], is_label: bool = False
    ):
        self.transitions = transitions
        attribute_name = "Next activity"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute, is_label)


class AttributeDataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"

    def __lt__(self, other):
        return self.value <= other.value


@dataclass(order=True)
class Attribute:
    major_attribute_type: MajorAttribute
    minor_attribute_type: MinorAttribute
    attribute_data_type: AttributeDataType
    df_attribute_name: str
    display_name: str
    query: str
    correlation: Optional[List[float]] = None
    p_val: Optional[List[float]] = None
    unit: Optional[str] = ""
    column_name: str = None  # for generic column attributes
    label_influence: Optional[List[float]] = None  # for categorical attributes only
    cases_with_attribute: Optional[int] = None  # for categorical attributes only
