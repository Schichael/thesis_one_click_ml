import abc
from enum import Enum


class MajorAttribute(Enum):
    ACTIVITY = "Activity"
    CASE = "Case"
    ordering = [ACTIVITY, CASE]

    def __lt__(self, other):
        return self.value <= other.value


class MinorAttribute(abc.ABC):
    def __init__(self, attribute_name: str, major_attribute: MajorAttribute):
        self.attribute_name = attribute_name
        self.major_attribute = major_attribute


class CaseDurationMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Case duration"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute)

class WorkInProgressMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Work in Progress"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute)

class EventCountMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Event count"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute)

class ReworkOccurenceMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Rework"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute)

class ActivityOccurenceMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Activity occurence"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute)

class EndActivityMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "End activity"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute)

class StartActivityMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Start activity"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute)

class ActivityTableColumnMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Activity table column"
        major_attribute = MajorAttribute.ACTIVITY
        super().__init__(attribute_name, major_attribute)

class CaseTableColumnMinorAttribute(MinorAttribute):
    def __init__(self):
        attribute_name = "Case table column"
        major_attribute = MajorAttribute.CASE
        super().__init__(attribute_name, major_attribute)


class AttributeDataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"

    def __lt__(self, other):
        return self.value <= other.value