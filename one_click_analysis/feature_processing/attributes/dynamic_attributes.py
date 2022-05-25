import abc
from typing import Optional

from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing.attributes.attribute import Attribute
from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDataType,
)
from one_click_analysis.feature_processing.attributes.attribute import AttributeType
from one_click_analysis.process_config.process_config import ProcessConfig


class DynamicAttribute(Attribute, abc.ABC):
    display_name = "Dynamic attribute"

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
        **kwargs,
    ):
        """

        :param process_config:
        :param attribute_name:
        :param pql_query:
        :param data_type:
        :param attribute_type:
        :param is_feature:
        :param is_class_feature:
        :param unit:
        :param column_name:
        :param kwargs: Additional arguments for Attribute base class
        """
        super().__init__(
            process_config=process_config,
            attribute_name=attribute_name,
            pql_query=pql_query,
            data_type=data_type,
            attribute_type=attribute_type,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            unit=unit,
            column_name=column_name,
            **kwargs,
        )


class NextActivityAttribute(DynamicAttribute):
    """The next activity"""

    display_name = "Next activity"
    description = "The activity that follows the current activity"

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
        attribute_name: str = "Next activity",
        **kwargs,
    ):
        self.process_config = process_config
        self.activity_table = self.process_config.table_dict[activity_table_str]
        self.attribute_name = attribute_name
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.OTHER,
            process_config=self.process_config,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            **kwargs,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            f'ACTIVITY_LEAD("{self.activity_table.table_str}".'
            f'"{self.activity_table.activity_col_str}", 1)'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class PreviousActivityColumnAttribute(DynamicAttribute):
    """Previous value of column in the Activity table"""

    display_name = "Previous categorical activity column value"
    description = (
        "The value of the specified activity table column at the event "
        "preceding the current event."
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        column_name: str,
        attribute_datatype: AttributeDataType,
        is_feature: bool = False,
        is_class_feature: bool = False,
        **kwargs,
    ):
        self.process_config = process_config
        self.activity_table = self.process_config.table_dict[activity_table_str]
        self.column_name = column_name
        self.attribute_name = (
            f"{self.activity_table.table_str}." f"{column_name} (" f"previous, dynamic)"
        )
        pql_query = self._gen_query()

        super().__init__(
            pql_query=pql_query,
            data_type=attribute_datatype,
            process_config=self.process_config,
            attribute_type=AttributeType.ACTIVITY_COL,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            column_name=column_name,
            **kwargs,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            f'ACTIVITY_LAG("{self.activity_table.table_str}".'
            f""
            f'"{self.column_name}", 1)'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class CurrentActivityColumnAttribute(DynamicAttribute):
    """Current value of column in the Activity table"""

    display_name = "Current categorical activity column value"
    description = (
        "The value of the specified activity table column at the current " "event"
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        column_name: str,
        attribute_datatype: AttributeDataType,
        is_feature: bool = False,
        is_class_feature: bool = False,
        **kwargs,
    ):
        self.process_config = process_config
        self.activity_table = self.process_config.table_dict[activity_table_str]
        self.column_name = column_name
        self.attribute_name = (
            f"{self.activity_table.table_str}." f"{column_name} (" f"current, dynamic)"
        )
        pql_query = self._gen_query()

        super().__init__(
            pql_query=pql_query,
            data_type=attribute_datatype,
            process_config=self.process_config,
            attribute_type=AttributeType.ACTIVITY_COL,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            column_name=column_name,
            **kwargs,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f'"{self.activity_table.table_str}".' f'"{self.column_name}"'
        return pql.PQLColumn(query=q, name=self.attribute_name)


class PreviousActivityOccurrenceAttribute(DynamicAttribute):
    """Number of times activity occurred in process before current row"""

    display_name = "Previous occurrence of activity"
    description = (
        "Checks whether or not an activity has occured before the current "
        "event (evaluates to 1 if yes and to 0 if no)"
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        activity: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
        **kwargs,
    ):
        self.process_config = process_config
        self.activity_table = self.process_config.table_dict[activity_table_str]
        self.activity = activity
        # Use implementation from prediction_builder
        self.attribute_name = f"Previous occurrence of activity {activity}"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.OTHER,
            process_config=self.process_config,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            **kwargs,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            f"CASE WHEN {self._get_query_string_without_case_table()} >= 1 THEN 1 ELSE "
            f"0 END"
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)

    def _get_query_string_without_case_table(self):
        return (
            f'COALESCE(RUNNING_SUM(CASE WHEN "{self.activity_table.table_str}"."'
            f"{self.activity_table.activity_col_str}\" = '{self.activity}' THEN 1 "
            f'ELSE 0 END, PARTITION BY ("{self.activity_table.table_str}"."'
            f'{self.activity_table.caseid_col_str}") ), 0)'
        )


class ActivityCountAttribute(DynamicAttribute):
    """Number of times activity occurred in process before current row"""

    display_name = "Activity count until current event"
    description = (
        "the number of times an activity has occured in the process before "
        "the current event"
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        activity: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
        **kwargs,
    ):
        self.process_config = process_config
        self.activity_table = self.process_config.table_dict[activity_table_str]
        self.activity = activity
        # Use implementation from prediction_builder
        self.attribute_name = f"Count of activity {activity}"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.OTHER,
            process_config=self.process_config,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            **kwargs,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f"""
            COALESCE(
                RUNNING_SUM(
                    CASE
                        WHEN "{self.activity_table.table_str}"."
                        {self.activity_table.activity_col_str}" =
                            '{self.activity}'
                        THEN 1
                        ELSE 0
                    END,
                    PARTITION BY ("
                    {self.activity_table.table_str}".
                    "{self.activity_table.caseid_col_str}") ),
            0)
        """
        return pql.PQLColumn(query=q, name=self.attribute_name)


class ActivityDurationAttribute(DynamicAttribute):
    """Duration of the current activity"""

    display_name = "Activity duration"
    description = (
        "The duration of the current activity. This is the difference in "
        "time between the previous and the current activity"
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
        unit: str = "DAYS",
        **kwargs,
    ):
        self.process_config = process_config
        self.activity_table = self.process_config.table_dict[activity_table_str]
        self.unit = unit
        self.attribute_name = f"Activity duration"

        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.OTHER,
            process_config=self.process_config,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            unit=self.unit.lower(),
            **kwargs,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f"""
            COALESCE(
                {self.unit}_BETWEEN(
                    ACTIVITY_LAG("{self.activity_table.table_str}".
                    "{self.activity_table.eventtime_col_str}", 1),
                    "{self.activity_table.table_str}".
                    "{self.activity_table.eventtime_col_str}"
                ), 0.0
            )
        """
        return pql.PQLColumn(query=q, name=self.attribute_name)


class DecisionToActivityAttribute(DynamicAttribute):
    """This attribute is specifically designed for the RoutingDecision analysis as
    the target attribute. A
    source activity A and target activities (e.g. T1, T2) are defined.
    If both T1 and T2 eventually follow A, only the first one is used. This attribute
    cannot be used without further processing since the PQL query only queries the
    current activity.
    """

    display_name = "Decision to Activity"
    description = (
        "The target activity that eventually follows the selected source activity. If "
        "multiple target activities are defined, it is the one that first eventually "
        "follows the source activity."
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
        **kwargs,
    ):
        self.process_config = process_config
        self.activity_table = self.process_config.table_dict[activity_table_str]
        self.attribute_name = f"Decision to activity"

        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.OTHER,
            process_config=self.process_config,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            **kwargs,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            f'"{self.activity_table.table_str}".'
            f'"{self.activity_table.activity_col_str}"'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)
