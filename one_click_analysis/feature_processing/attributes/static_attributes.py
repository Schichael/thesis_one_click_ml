import abc
from typing import Optional

from prediction_builder.data_extraction import ProcessModel
from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing.attributes import attribute_utils
from one_click_analysis.feature_processing.attributes.attribute import Attribute
from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDataType,
)
from one_click_analysis.feature_processing.attributes.attribute import AttributeType


class StaticAttribute(Attribute, abc.ABC):
    display_name = "Static attribute"

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
        super().__init__(
            process_model=process_model,
            attribute_name=attribute_name,
            pql_query=pql_query,
            data_type=data_type,
            attribute_type=attribute_type,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            unit=unit,
            column_name=column_name,
        )

    def get_query_with_value(self, value: Optional[str] = None):
        """Get PQL query for the attribute for a specific value. If the
        attribute was one-hot-encoded, the query for just the value is used. The
        result will be 0s and 1s. If the attribute was not one-hot-encoded, the
        pql query is the same as for the attribute.

        :param value: specific value of the attribute (from one-hot-encoding)
        :return: the PQL query
        """
        return self.pql_query


class CaseDurationAttribute(StaticAttribute):
    """Duration of the whle case"""

    display_name = "Case duration"

    def __init__(
        self,
        process_model: ProcessModel,
        time_aggregation: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.time_aggregation = time_aggregation
        self.attribute_name = "Case duration"
        pql_query = self._gen_query()
        super().__init__(
            process_model=process_model,
            attribute_name=self.attribute_name,
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.OTHER,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            unit=self.time_aggregation.lower(),
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            "(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO ALL_OCCURRENCE["
            "'Process End'], REMAP_TIMESTAMPS(\""
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.timestamp_column_str
            + '", '
            "" + self.time_aggregation + ")))"
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class WorkInProgressAttribute(StaticAttribute):
    """Work in Progress for whole case"""

    display_name = "Work in Progress during case"

    def __init__(
        self,
        process_model: ProcessModel,
        aggregation: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.aggregation = aggregation
        aggregation_df_name = attribute_utils.get_aggregation_df_name(aggregation)
        self.attribute_name = "Case Work in progress" + " (" + aggregation_df_name + ")"
        pql_query = self._gen_query()
        super().__init__(
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.OTHER,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            "PU_"
            + self.aggregation
            + ' ( "'
            + self.process_model.case_table_str
            + '", '
            "RUNNING_SUM( "
            "CASE WHEN "
            'INDEX_ACTIVITY_ORDER ( "'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.activity_column_str
            + '" ) = 1 THEN 1'
            " WHEN "
            'INDEX_ACTIVITY_ORDER_REVERSE ( "'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.activity_column_str
            + '" ) = 1 THEN -1 ELSE 0 END, ORDER BY ( "'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.timestamp_column_str
            + '" ) ) )'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class EventCountAttribute(StaticAttribute):
    """Event Count for a whole case"""

    display_name = "Case Event count"

    def __init__(
        self,
        process_model: ProcessModel,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.attribute_name = "Case Event count"
        pql_query = self._gen_query()
        super().__init__(
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.OTHER,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            'PU_COUNT("' + self.process_model.case_table_str + '", '
            '"'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.activity_column_str
            + '")'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class ActivityOccurenceAttribute(StaticAttribute):
    """Rework occurence for a whole case"""

    display_name = "Activity occurence in case"

    def __init__(
        self,
        process_model: ProcessModel,
        activity: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.activity = activity
        self.attribute_name = f"Activity = {self.activity} (occurence)"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            attribute_type=AttributeType.OTHER,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            f'CASE WHEN PU_SUM("{self.process_model.case_table_str}", CASE WHEN "'
            f'{self.process_model.activity_table_str}"."'
            f"{self.process_model.activity_column_str}\" = '{self.activity}' THEN 1 "
            f"ELSE 0"
            f" END) >= 1 THEN 1 ELSE 0 END"
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class ReworkCountAttribute(StaticAttribute):
    """Count of reworked activities"""

    display_name = "Rework count in case"

    def __init__(
        self,
        process_model: ProcessModel,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.attribute_name = f"Case Rework count (all activities)"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f"""
            PU_SUM("{self.process_model.case_table_str}",
                CASE
                    WHEN INDEX_ACTIVITY_TYPE (
                        {self.process_model.activity_column.query}) > 1
                    THEN 1
                    ELSE 0
                END)
            """
        return pql.PQLColumn(query=q, name=self.attribute_name)


class ReworkOccurrenceAttribute(StaticAttribute):
    """Whether any activity was done more than once"""

    display_name = "Rework occurence in case"

    def __init__(
        self,
        process_model: ProcessModel,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.attribute_name = f"Case Rework occurence (any activity)"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f"""
            CASE
            WHEN PU_SUM("{self.process_model.case_table_str}",
                CASE
                    WHEN INDEX_ACTIVITY_TYPE (
                        {self.process_model.activity_column.query}) > 1
                    THEN 1
                    ELSE 0
                END) >=1 THEN 1 ELSE 0 END
            """
        return pql.PQLColumn(query=q, name=self.attribute_name)


class StartActivityAttribute(StaticAttribute):
    """Start activity"""

    display_name = "Start activity"

    def __init__(
        self,
        process_model: ProcessModel,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.attribute_name = f"Start activity"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            'PU_FIRST("' + self.process_model.case_table_str + '", '
            '"'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.activity_column_str
            + '")'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)

    def get_query_with_value(self, value: Optional[str] = None):
        """Get PQL query for the attribute for a specific value. If the
        attribute was one-hot-encoded, the query for just the value is used. The
        result will be 0s and 1s. If the attribute was not one-hot-encoded, the
        pql query is the same as for the attribute.

        :param value: specific value of the attribute (from one-hot-encoding)
        :return: the PQL query
        """
        q = (
            f'CASE WHEN  PU_FIRST("{self.process_model.case_table_str}", '
            + f'"{self.process_model.activity_table_str}"."'
            f"{self.process_model.activity_column_str}\" = '{value}') THEN 1 ELSE "
            f"0 END"
        )

        return pql.PQLColumn(name=f"{self.attribute_name} = {value}", query=q)


class EndActivityAttribute(StaticAttribute):
    """End activity"""

    display_name = "End activity"

    def __init__(
        self,
        process_model: ProcessModel,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.attribute_name = f"End activity"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            'PU_LAST("' + self.process_model.case_table_str + '", '
            '"'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.activity_column_str
            + '")'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)

    def get_query_with_value(self, value: Optional[str] = None):
        """Get PQL query for the attribute for a specific value. If the
        attribute was one-hot-encoded, the query for just the value is used. The
        result will be 0s and 1s. If the attribute was not one-hot-encoded, the
        pql query is the same as for the attribute.

        :param value: specific value of the attribute (from one-hot-encoding)
        :return: the PQL query
        """
        q = (
            f'CASE WHEN  PU_LAST("{self.process_model.case_table_str}", '
            + f'"{self.process_model.activity_table_str}"."'
            f"{self.process_model.activity_column_str}\" = '{value}') THEN 1 ELSE "
            f"0 END"
        )

        return pql.PQLColumn(name=f"{self.attribute_name} = {value}", query=q)


class NumericActivityTableColumnAttribute(StaticAttribute):
    """Any numeric activity table column."""

    display_name = "Numeric activity table column aggregation"

    def __init__(
        self,
        process_model: ProcessModel,
        column_name: str,
        aggregation: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.column_name = column_name
        self.aggregation = aggregation  # aggregation for PU function
        self.attribute_name = (
            f"{self.process_model.activity_table_str}."
            f"{self.column_name} ("
            f"{attribute_utils.get_aggregation_df_name(aggregation)})"
        )
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.ACTIVITY_COL_NUMERICAL,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            column_name=column_name,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            "PU_" + self.aggregation + '("' + self.process_model.case_table_str + '", '
            '"'
            + self.process_model.activity_table_str
            + '"."'
            + self.column_name
            + '")'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class CaseTableColumnNumericAttribute(StaticAttribute):
    """Any case table column."""

    display_name = "Numeric case table column"

    def __init__(
        self,
        process_model: ProcessModel,
        column_name: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.column_name = column_name
        self.attribute_name = (
            f"{self.process_model.case_table_str}." f"{self.column_name}"
        )
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.CASE_COL_NUMERICAL,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            column_name=column_name,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f'"{self.process_model.case_table_str}"."{self.column_name}"'
        return pql.PQLColumn(query=q, name=self.attribute_name)


class CaseTableColumnCategoricalAttribute(StaticAttribute):
    """Any case table column."""

    display_name = "Categorical case table column"

    def __init__(
        self,
        process_model: ProcessModel,
        column_name: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.column_name = column_name
        self.attribute_name = (
            f"{self.process_model.case_table_str}." f"{self.column_name}"
        )
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.CASE_COL_CATEGORICAL,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            column_name=column_name,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f'"{self.process_model.case_table_str}"."{self.column_name}"'
        return pql.PQLColumn(query=q, name=self.attribute_name)

    def get_query_with_value(self, value: Optional[str] = None):
        """Get PQL query for the attribute for a specific value. If the
        attribute was one-hot-encoded, the query for just the value is used. The
        result will be 0s and 1s. If the attribute was not one-hot-encoded, the
        pql query is the same as for the attribute.

        :param value: specific value of the attribute (from one-hot-encoding)
        :return: the PQL query
        """
        q = (
            f'CASE WHEN "{self.process_model.case_table_str}"."{self.column_name}" = '
            f"'value' THEN 1 ELSE 0 END"
        )

        return pql.PQLColumn(name=f"{self.attribute_name} = {value}", query=q)


class TransitionOccurenceAttribute(StaticAttribute):
    """Whether a transition happens in a case"""

    display_name = "Transition occurence in case"

    def __init__(
        self,
        process_model: ProcessModel,
        transition_start: str,
        transition_end: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.transition_start = transition_start
        self.transition_end = transition_end
        self.attribute_name = (
            f"Transition occurence ({transition_start} -> {transition_end})"
        )
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            f"CASE WHEN PROCESS EQUALS '{self.transition_start}' TO "
            f"'{self.transition_end}' THEN 1 ELSE 0 END"
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class StartActivityTimeAttribute(StaticAttribute):
    """Start activity time"""

    display_name = "Start activity time"

    def __init__(self, process_model: ProcessModel):
        self.process_model = process_model
        self.attribute_name = f"Start activity Time"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.DATETIME,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=False,
            is_class_feature=False,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            'PU_FIRST("' + self.process_model.case_table_str + '", '
            '"'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.timestamp_column_str
            + '")'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class EndActivityTimeAttribute(StaticAttribute):
    """End activity time"""

    display_name = "End activity time"

    def __init__(self, process_model: ProcessModel):
        self.process_model = process_model
        self.attribute_name = f"End activity Time"
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.DATETIME,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=False,
            is_class_feature=False,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            'PU_Last("' + self.process_model.case_table_str + '", '
            '"'
            + self.process_model.activity_table_str
            + '"."'
            + self.process_model.timestamp_column_str
            + '")'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)

    def gen_query_with_value(self, value: Optional[str] = None):
        return self.pql_query
