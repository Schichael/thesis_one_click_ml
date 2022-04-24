import abc
from typing import Optional

from prediction_builder.data_extraction import dynamic_features
from prediction_builder.data_extraction import ProcessModel
from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing.attributes_new.attribute import Attribute
from one_click_analysis.feature_processing.attributes_new.attribute import (
    AttributeDataType,
)
from one_click_analysis.feature_processing.attributes_new.attribute import AttributeType


class DynamicAttribute(Attribute, abc.ABC):
    display_name = "Dynamic attribute"

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


class NextActivityAttribute(DynamicAttribute):
    """The next activity"""

    display_name = "Next activity"

    def __init__(
        self,
        process_model: ProcessModel,
        is_feature: bool = False,
        is_class_feature: bool = False,
        attribute_name: str = "Next activity",
    ):
        self.process_model = process_model
        self.attribute_name = attribute_name
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
            f'ACTIVITY_LEAD("{self.process_model.activity_table_str}".'
            f'"{self.process_model.activity_column_str}", 1)'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class CurrentNumericalActivityColumnAttribute(DynamicAttribute):
    """Current value of column in the Activity table"""

    display_name = "Current numerical activity column value"

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
            f"{self.process_model.activity_table_str}." f"{column_name} (dynamic)"
        )
        pql_query = self._gen_query()

        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            process_model=self.process_model,
            attribute_type=AttributeType.ACTIVITY_COL_NUMERICAL,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            column_name=column_name,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f'"{self.process_model.activity_table_str}".' f'"{self.column_name}"'
        return pql.PQLColumn(query=q, name=self.attribute_name)


class CurrentCategoricalActivityColumnAttribute(DynamicAttribute):
    """Current value of column in the Activity table"""

    display_name = "Current categorical activity column value"

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
            f"{self.process_model.activity_table_str}." f"{column_name} (dynamic)"
        )
        pql_query = self._gen_query()

        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
            process_model=self.process_model,
            attribute_type=AttributeType.ACTIVITY_COL_CATEGORICAL,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            column_name=column_name,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = f'"{self.process_model.activity_table_str}".' f'"{self.column_name}"'
        return pql.PQLColumn(query=q, name=self.attribute_name)


class ActivityCountAttribute(DynamicAttribute):
    """Number of times activity occurred in process before current row"""

    display_name = "Activity count until current event"

    def __init__(
        self,
        process_model: ProcessModel,
        activity: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.activity = activity
        # Use implementation from prediction_builder
        self.dyn_feature = dynamic_features.ActivityCount(process_model, activity)
        self.attribute_name = f"Count of activity {activity}"
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
        q = self.dyn_feature.query
        return pql.PQLColumn(query=q, name=self.attribute_name)


class ActivityDurationAttribute(DynamicAttribute):
    """Duration of the current activity"""

    display_name = "Activity duration"

    def __init__(
        self,
        process_model: ProcessModel,
        is_feature: bool = False,
        is_class_feature: bool = False,
        unit: str = "DAYS",
    ):
        self.process_model = process_model
        # Use implementation from prediction_builder
        self.unit = unit
        self.dyn_feature = dynamic_features.ActivityDuration(process_model, unit=unit)
        self.attribute_name = f"Activity duration"

        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.NUMERICAL,
            attribute_type=AttributeType.OTHER,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            unit=unit,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = self.dyn_feature.query
        return pql.PQLColumn(query=q, name=self.attribute_name)
