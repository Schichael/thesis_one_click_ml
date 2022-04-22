import abc

from prediction_builder.data_extraction import dynamic_features
from prediction_builder.data_extraction import ProcessModel
from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing.attributes_new.attribute import Attribute
from one_click_analysis.feature_processing.attributes_new.attribute import (
    AttributeDataType,
)


class DynamicAttribute(Attribute, abc.ABC):
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
        super().__init__(
            process_model=process_model,
            attribute_name=attribute_name,
            pql_query=pql_query,
            data_type=data_type,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            unit=unit,
        )


class NextActivityAttribute(DynamicAttribute):
    """The next activity"""

    def __init__(
        self,
        process_model: ProcessModel,
        next_activity: str,
        is_feature: bool = False,
        is_class_feature: bool = False,
        attribute_name: str = "Next activity",
    ):
        self.process_model = process_model
        self.next_activity = next_activity
        self.attribute_name = attribute_name
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=AttributeDataType.CATEGORICAL,
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


class CurrentActivityColumnAttribute(DynamicAttribute):
    """Current value of column in the Activity table"""

    def __init__(
        self,
        process_model: ProcessModel,
        column_name: str,
        data_type: AttributeDataType,
        is_feature: bool = False,
        is_class_feature: bool = False,
    ):
        self.process_model = process_model
        self.column_name = column_name
        self.attribute_name = (
            f"{self.process_model.activity_table_str}."
            f"{self.process_model.activity_column_str}"
        )
        pql_query = self._gen_query()
        super().__init__(
            pql_query=pql_query,
            data_type=data_type,
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = (
            f'"{self.process_model.activity_table_str}".'
            f'"{self.process_model.activity_column_str}"'
        )
        return pql.PQLColumn(query=q, name=self.attribute_name)


class ActivityCountAttribute(DynamicAttribute):
    """Number of times activity occurred in process before current row"""

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
            process_model=self.process_model,
            attribute_name=self.attribute_name,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
        )

    def _gen_query(self) -> pql.PQLColumn:
        q = self.dyn_feature.query
        return pql.PQLColumn(query=q, name=self.attribute_name)
