import abc
import timeit
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing import feature_processor_new
from one_click_analysis.feature_processing import post_processing
from one_click_analysis.feature_processing.attributes import dynamic_attributes
from one_click_analysis.feature_processing.attributes import static_attributes
from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDescriptor,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    DecisionToActivityAttribute,
)  # noqa
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    PreviousActivityOccurrenceAttribute,
)  # noqa
from one_click_analysis.feature_processing.post_processing import PostProcessor
from one_click_analysis.feature_processing.statistics_computer import StatisticsComputer
from one_click_analysis.process_config.process_config import ProcessConfig


class UseCaseProcessor(abc.ABC):
    def __init__(
        self,
        process_config: ProcessConfig,
        used_static_attribute_descriptors: List[AttributeDescriptor],
        used_dynamic_attribute_descriptors: List[AttributeDescriptor],
        target_attribute_descriptor: AttributeDescriptor,
        considered_activity_table_cols: List[str],
        considered_case_level_table_cols: Dict[str, List[str]],
        is_closed_query: pql.PQLColumn,
        **kwargs,
    ):
        self.process_config = process_config
        self.used_static_attribute_descriptors = used_static_attribute_descriptors
        self.used_dynamic_attribute_descriptors = used_dynamic_attribute_descriptors
        self.target_attribute_descriptor = target_attribute_descriptor
        self.considered_activity_table_cols = considered_activity_table_cols
        self.considered_case_level_table_cols = considered_case_level_table_cols
        self.is_closed_query = is_closed_query
        self.is_closed_filter = self._create_is_closed_filter(self.is_closed_query)
        self.chunksize = kwargs.get("chunksize", 10000)
        self.min_attr_count_perc = kwargs.get("min_attr_count_perc", 0.02)
        self.max_attr_count_perc = kwargs.get("max_attr_count_perc", 0.98)
        self.th_remove_col = kwargs.get("th_remove_col", 0.3)
        self.df_x = None
        self.df_target = None
        self.target_features = None
        self.features = None
        self.filters = []
        self.filters.append(self.is_closed_filter)
        self.df_timestamp_column = None
        self.num_cases = None

    def _create_is_closed_filter(self, is_closed_query: pql.PQLColumn) -> pql.PQLFilter:
        """Create IS_CLOSED PQLFilter object"""
        query_str = is_closed_query.query
        query_str_filter = query_str + " = 1"
        return pql.PQLFilter(query=query_str_filter)

    @abc.abstractmethod
    def process(self):
        pass


class CaseDurationProcessor(UseCaseProcessor):
    """Feature Processor for the case duration use case."""

    # attributes that can be used for this use case
    work_in_progress_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.WorkInProgressAttribute,
        display_name=static_attributes.WorkInProgressAttribute.display_name,
        description=static_attributes.WorkInProgressAttribute.description,
    )

    start_activity_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.StartActivityAttribute,
        display_name=static_attributes.StartActivityAttribute.display_name,
        description=static_attributes.StartActivityAttribute.description,
    )

    end_activity_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.EndActivityAttribute,
        display_name=static_attributes.EndActivityAttribute.display_name,
        description=static_attributes.EndActivityAttribute.description,
    )

    activity_occurrence_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.ActivityOccurenceAttribute,
        display_name=static_attributes.ActivityOccurenceAttribute.display_name,
        description=static_attributes.ActivityOccurenceAttribute.description,
    )

    numeric_activity_table_col_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.NumericActivityTableColumnAttribute,
        display_name=static_attributes.NumericActivityTableColumnAttribute.display_name,
        description=static_attributes.NumericActivityTableColumnAttribute.description,
    )

    case_table_col_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.CaseTableColumnAttribute,
        display_name=static_attributes.CaseTableColumnAttribute.display_name,
        description=static_attributes.CaseTableColumnAttribute.description,
    )

    potential_static_attributes_descriptors = [
        start_activity_attr_descriptor,
        end_activity_attr_descriptor,
        activity_occurrence_attr_descriptor,
        work_in_progress_attr_descriptor,
        numeric_activity_table_col_attr_descriptor,
        case_table_col_attr_descriptor,
    ]
    potential_dynamic_attributes_descriptors = []
    # the target attribute that is used for this use case
    target_attribute_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.CaseDurationAttribute,
        display_name=static_attributes.CaseDurationAttribute.display_name,
        description=static_attributes.CaseDurationAttribute.description,
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        used_static_attribute_descriptors: List[AttributeDescriptor],
        used_dynamic_attribute_descriptors: List[AttributeDescriptor],
        considered_activity_table_cols: List[str],
        considered_case_level_table_cols: Dict[str, List[str]],
        is_closed_query: pql.PQLColumn,
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            process_config=process_config,
            used_static_attribute_descriptors=used_static_attribute_descriptors,
            used_dynamic_attribute_descriptors=used_dynamic_attribute_descriptors,
            target_attribute_descriptor=self.target_attribute_descriptor,
            considered_activity_table_cols=considered_activity_table_cols,
            considered_case_level_table_cols=considered_case_level_table_cols,
            is_closed_query=is_closed_query,
            **kwargs,
        )
        self.activity_table_str = activity_table_str
        self.time_unit = time_unit
        self.start_date = start_date
        self.end_date = end_date

    def process(self):
        date_filters = feature_processor_new.date_filter_PQL(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.filters = self.filters + date_filters

        self.num_cases = feature_processor_new.get_number_cases(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            filters=self.filters,
        )

        (
            min_attr_count,
            max_attr_count,
        ) = feature_processor_new.compute_min_max_attribute_counts_PQL(
            self.min_attr_count_perc,
            self.max_attr_count_perc,
            process_config=self.process_config,
            activity_table_name=self.activity_table_str,
            chunksize=self.chunksize,
            filters=self.filters,
        )

        used_static_attributes = self._gen_static_attr_list(
            min_attr_count=min_attr_count, max_attr_count=max_attr_count
        )

        self.df_timestamp_column = "Start activity Time"

        target_attribute = static_attributes.CaseDurationAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            time_aggregation=self.time_unit,
            is_feature=False,
            is_class_feature=True,
        )

        # Define a default target variable
        target_variable = target_attribute.pql_query

        # Get DataFrames
        self.df_x, self.df_target = feature_processor_new.extract_dfs(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            static_attributes=used_static_attributes,
            dynamic_attributes=[],
            is_closed_indicator=self.is_closed_query,
            target_variable=target_variable,
            filters=self.filters,
        )

        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=used_static_attributes,
            target_attributes=target_attribute,
            valid_target_values=None,
            invalid_target_replacement=None,
            min_counts_perc=self.min_attr_count_perc,
            max_counts_perc=self.max_attr_count_perc,
        )

        self.df_x, self.df_target, self.target_features, self.features = pp.process()

        statistics_computer = StatisticsComputer(
            features=self.features,
            target_features=self.target_features,
            df_x=self.df_x,
            df_target=self.df_target,
        )
        statistics_computer.compute_all_statistics()
        self.df_x, self.df_target = post_processing.remove_nan(
            df_x=self.df_x,
            df_target=self.df_target,
            features=self.features,
            target_features=self.target_features,
            th_remove_col=self.th_remove_col,
        )

    def _gen_static_attr_list(self, min_attr_count: int, max_attr_count: int):
        """Gen list of used static attributes."""
        static_attributes_list = []
        # Add attributes that always need to be added for processing
        static_attributes_list.append(
            static_attributes.StartActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )
        static_attributes_list.append(
            static_attributes.EndActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )

        # Add used attributes
        if (
            self.work_in_progress_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            static_attributes_list.append(
                static_attributes.WorkInProgressAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    aggregation="AVG",
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if (
            self.start_activity_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            static_attributes_list.append(
                static_attributes.StartActivityAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if self.end_activity_attr_descriptor in self.used_static_attribute_descriptors:
            static_attributes_list.append(
                static_attributes.EndActivityAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if (
            self.activity_occurrence_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            activity_occ_attributes = (
                feature_processor_new.gen_static_activity_occurence_attributes(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    is_feature=True,
                    min_vals=min_attr_count,
                    max_vals=max_attr_count,
                )
            )
            static_attributes_list = static_attributes_list + activity_occ_attributes

        if (
            self.numeric_activity_table_col_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            numeric_activity_attirbutes = (
                feature_processor_new.gen_static_numeric_activity_table_attributes(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    columns=self.considered_activity_table_cols,
                    aggregation="AVG",
                    is_feature=True,
                    is_class_feature=False,
                )
            )
            static_attributes_list = (
                static_attributes_list + numeric_activity_attirbutes
            )

        if (
            self.case_table_col_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            case_table_col_attirbutes = (
                feature_processor_new.gen_case_column_attributes_multi_table(
                    process_config=self.process_config,
                    table_column_dict=self.considered_case_level_table_cols,
                    is_feature=True,
                    is_class_feature=False,
                )
            )
            static_attributes_list = static_attributes_list + case_table_col_attirbutes

        return static_attributes_list


# case_table_col_attr_descriptor


class TransitionTimeProcessor(UseCaseProcessor):
    """Feature Processor for the transition time (bottle neck) use case."""

    # attributes that can be used for this use case
    start_activity_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.StartActivityAttribute,
        display_name=static_attributes.StartActivityAttribute.display_name,
        description=static_attributes.StartActivityAttribute.description,
    )

    case_table_col_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.NumericActivityTableColumnAttribute,
        display_name=static_attributes.NumericActivityTableColumnAttribute.display_name,
        description=static_attributes.NumericActivityTableColumnAttribute.description,
    )

    current_activity_col_attr_descriptor = AttributeDescriptor(
        attribute_type=dynamic_attributes.CurrentActivityColumnAttribute,
        display_name=dynamic_attributes.CurrentActivityColumnAttribute.display_name,
        description=dynamic_attributes.CurrentActivityColumnAttribute.description,
    )

    previous_activity_occurrence_attr_descriptor = AttributeDescriptor(
        attribute_type=PreviousActivityOccurrenceAttribute,
        display_name=PreviousActivityOccurrenceAttribute.display_name,
        description=PreviousActivityOccurrenceAttribute.description,
    )
    # PreviousActivityOccurrenceAttribute
    potential_static_attributes_descriptors = [
        start_activity_attr_descriptor,
        case_table_col_attr_descriptor,
    ]
    potential_dynamic_attributes_descriptors = [
        current_activity_col_attr_descriptor,
        previous_activity_occurrence_attr_descriptor,
    ]
    # the target attribute that is used for this use case
    target_attribute_descriptor = AttributeDescriptor(
        attribute_type=dynamic_attributes.ActivityDurationAttribute,
        display_name=dynamic_attributes.ActivityDurationAttribute.display_name,
        description=dynamic_attributes.ActivityDurationAttribute.description,
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        used_static_attribute_descriptors: List[AttributeDescriptor],
        used_dynamic_attribute_descriptors: List[AttributeDescriptor],
        considered_activity_table_cols: List[str],
        considered_case_level_table_cols: Dict[str, List[str]],
        is_closed_query: pql.PQLColumn,
        source_activity: str,
        target_activity: str,
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            process_config=process_config,
            used_static_attribute_descriptors=used_static_attribute_descriptors,
            used_dynamic_attribute_descriptors=used_dynamic_attribute_descriptors,
            target_attribute_descriptor=self.target_attribute_descriptor,
            considered_activity_table_cols=considered_activity_table_cols,
            considered_case_level_table_cols=considered_case_level_table_cols,
            is_closed_query=is_closed_query,
            **kwargs,
        )
        self.activity_table_str = activity_table_str
        self.time_unit = time_unit
        self.start_date = start_date
        self.end_date = end_date
        self.source_activity = source_activity
        self.target_activity = target_activity

    def process(self):
        date_filters = feature_processor_new.date_filter_PQL(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.filters = self.filters + date_filters

        self.num_cases = feature_processor_new.get_number_cases(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            filters=self.filters,
        )

        prev_activity_filter = feature_processor_new.filter_prev_activity(
            prev_activity=self.source_activity,
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
        )

        self.filters = self.filters + [prev_activity_filter]

        (
            min_attr_count,
            max_attr_count,
        ) = feature_processor_new.compute_min_max_attribute_counts_PQL(
            self.min_attr_count_perc,
            self.max_attr_count_perc,
            process_config=self.process_config,
            activity_table_name=self.activity_table_str,
            chunksize=self.chunksize,
            filters=self.filters,
        )

        used_static_attributes = self._gen_static_attr_list(
            min_attr_count=min_attr_count, max_attr_count=max_attr_count
        )
        used_dynamic_attributes = self._gen_dynamic_attr_list(
            min_attr_count=min_attr_count, max_attr_count=max_attr_count
        )

        self.df_timestamp_column = "Start activity Time"

        target_attribute = dynamic_attributes.ActivityDurationAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            unit=self.time_unit,
            is_feature=False,
            is_class_feature=True,
        )

        # Define a default target variable
        target_variable = target_attribute.pql_query
        key_activities = [self.target_activity]
        # Get DataFrames. Need to add [target_attribute] to dynamic_attributes in case
        # that there are no dynamic features. Then one would get an error.
        target_attribute_for_dyn = dynamic_attributes.ActivityDurationAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            unit=self.time_unit,
            is_feature=False,
            is_class_feature=False,
        )
        target_attribute_for_dyn.attribute_name = "temp_attr"
        self.df_x, self.df_target = feature_processor_new.extract_dfs(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            key_activities=key_activities,
            static_attributes=used_static_attributes,
            dynamic_attributes=used_dynamic_attributes + [target_attribute_for_dyn],
            is_closed_indicator=self.is_closed_query,
            target_variable=target_variable,
            filters=self.filters,
        )

        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=used_static_attributes,
            target_attributes=target_attribute,
            valid_target_values=None,
            invalid_target_replacement=None,
            min_counts_perc=self.min_attr_count_perc,
            max_counts_perc=self.max_attr_count_perc,
        )

        self.df_x, self.df_target, self.target_features, self.features = pp.process()

        statistics_computer = StatisticsComputer(
            features=self.features,
            target_features=self.target_features,
            df_x=self.df_x,
            df_target=self.df_target,
        )
        statistics_computer.compute_all_statistics()
        self.df_x, self.df_target = post_processing.remove_nan(
            df_x=self.df_x,
            df_target=self.df_target,
            features=self.features,
            target_features=self.target_features,
            th_remove_col=self.th_remove_col,
        )

    def _gen_static_attr_list(self, min_attr_count: int, max_attr_count: int):
        """Gen list of used static attributes."""
        static_attributes_list = []
        # Add attributes that always need to be added for processing
        static_attributes_list.append(
            static_attributes.StartActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )
        static_attributes_list.append(
            static_attributes.EndActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )

        # Add used attributes

        if (
            self.start_activity_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            static_attributes_list.append(
                static_attributes.StartActivityAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if (
            self.case_table_col_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            case_table_col_attirbutes = (
                feature_processor_new.gen_case_column_attributes_multi_table(
                    process_config=self.process_config,
                    table_column_dict=self.considered_case_level_table_cols,
                    is_feature=True,
                    is_class_feature=False,
                )
            )
            static_attributes_list = static_attributes_list + case_table_col_attirbutes

        return static_attributes_list

    def _gen_dynamic_attr_list(self, min_attr_count: int, max_attr_count: int):
        """Gen list of used static attributes."""
        dynamic_attributes_list = []

        # Add used attributes

        if (
            self.current_activity_col_attr_descriptor
            in self.used_dynamic_attribute_descriptors
        ):
            current_activity_col_attributes = (
                feature_processor_new.gen_current_activity_col_attributes(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    columns=self.considered_activity_table_cols,
                    is_feature=True,
                    is_class_feature=False,
                )
            )
            dynamic_attributes_list = (
                dynamic_attributes_list + current_activity_col_attributes
            )

        if (
            self.previous_activity_occurrence_attr_descriptor
            in self.used_dynamic_attribute_descriptors
        ):
            previous_activity_occ_attributes = (
                feature_processor_new.gen_dynamic_activity_occurence_attributes(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    activities=None,
                    min_vals=min_attr_count,
                    max_vals=max_attr_count,
                    is_feature=True,
                    is_class_feature=False,
                    filters=self.filters,
                )
            )
            dynamic_attributes_list = (
                dynamic_attributes_list + previous_activity_occ_attributes
            )

        return dynamic_attributes_list


class RoutingDecisionProcessor(UseCaseProcessor):
    """Feature Processor for the transition time (bottle neck) use case."""

    # attributes that can be used for this use case
    start_activity_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.StartActivityAttribute,
        display_name=static_attributes.StartActivityAttribute.display_name,
        description=static_attributes.StartActivityAttribute.description,
    )

    case_table_col_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.CaseTableColumnAttribute,
        display_name=static_attributes.CaseTableColumnAttribute.display_name,
        description=static_attributes.CaseTableColumnAttribute.description,
    )

    current_activity_col_attr_descriptor = AttributeDescriptor(
        attribute_type=dynamic_attributes.CurrentActivityColumnAttribute,
        display_name="Activity table column value during source activity",
        description="The values of an activity table column at the time the source "
        "activity occurs",
    )

    previous_activity_occurrence_attr_descriptor = AttributeDescriptor(
        attribute_type=PreviousActivityOccurrenceAttribute,
        display_name=PreviousActivityOccurrenceAttribute.display_name,
        description="Checks whether or not an activity has occured before the source "
        "activity (evaluates to 1 if yes and to 0 if no)",
    )
    # PreviousActivityOccurrenceAttribute
    potential_static_attributes_descriptors = [
        start_activity_attr_descriptor,
        case_table_col_attr_descriptor,
    ]
    potential_dynamic_attributes_descriptors = [
        current_activity_col_attr_descriptor,
        previous_activity_occurrence_attr_descriptor,
    ]

    # the target attribute that is used for this use case
    target_attribute_descriptor = AttributeDescriptor(
        attribute_type=dynamic_attributes.DecisionToActivityAttribute,
        display_name=dynamic_attributes.DecisionToActivityAttribute.display_name,
        description=dynamic_attributes.DecisionToActivityAttribute.description,
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        used_static_attribute_descriptors: List[AttributeDescriptor],
        used_dynamic_attribute_descriptors: List[AttributeDescriptor],
        considered_activity_table_cols: List[str],
        considered_case_level_table_cols: Dict[str, List[str]],
        is_closed_query: pql.PQLColumn,
        source_activity: str,
        target_activities: List[str],
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            process_config=process_config,
            used_static_attribute_descriptors=used_static_attribute_descriptors,
            used_dynamic_attribute_descriptors=used_dynamic_attribute_descriptors,
            target_attribute_descriptor=self.target_attribute_descriptor,
            considered_activity_table_cols=considered_activity_table_cols,
            considered_case_level_table_cols=considered_case_level_table_cols,
            is_closed_query=is_closed_query,
            **kwargs,
        )
        self.activity_table_str = activity_table_str
        self.time_unit = time_unit
        self.start_date = start_date
        self.end_date = end_date
        self.source_activity = source_activity
        self.target_activities = target_activities
        self.case_duration_attribute: Optional[
            static_attributes.CaseDurationAttribute
        ] = None

    def process(self):
        date_filters = feature_processor_new.date_filter_PQL(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.filters = self.filters + date_filters

        is_closed_indicator = feature_processor_new.all_cases_closed_query(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            name="IS_CLOSED",
        )

        self.num_cases = feature_processor_new.get_number_cases(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            filters=self.filters,
        )

        (
            min_attr_count,
            max_attr_count,
        ) = feature_processor_new.compute_min_max_attribute_counts_PQL(
            self.min_attr_count_perc,
            self.max_attr_count_perc,
            process_config=self.process_config,
            activity_table_name=self.activity_table_str,
            chunksize=self.chunksize,
            filters=self.filters,
        )

        used_static_attributes = self._gen_static_attr_list(
            min_attr_count=min_attr_count, max_attr_count=max_attr_count
        )
        used_dynamic_attributes = self._gen_dynamic_attr_list(
            min_attr_count=min_attr_count, max_attr_count=max_attr_count
        )

        self.df_timestamp_column = "Start activity Time"

        target_attribute = dynamic_attributes.DecisionToActivityAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            is_feature=False,
            is_class_feature=True,
        )

        # First get features with key activity being only the source activity. This
        # defines only the features and not yet the target features. This is done to
        # save memory As the real feature can result in a multiple of the rows that
        # are needed for the features.
        target_variable = target_attribute.pql_query
        key_activities = [self.source_activity]
        # Get DataFrames. Need to add [target_attribute] to dynamic_attributes in case
        # that there are no dynamic features. Then one would get an error.
        target_attribute_for_dyn = dynamic_attributes.ActivityDurationAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            unit=self.time_unit,
            is_feature=False,
            is_class_feature=False,
        )
        target_attribute_for_dyn.attribute_name = "temp_attr"
        df_x, _ = feature_processor_new.extract_dfs(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            key_activities=key_activities,
            static_attributes=used_static_attributes,
            dynamic_attributes=used_dynamic_attributes + [target_attribute_for_dyn],
            is_closed_indicator=is_closed_indicator,
            target_variable=target_variable,
            filters=self.filters,
        )

        # Now get the real target features.
        key_activities = [self.source_activity] + self.target_activities
        # Get DataFrame. Need to add [target_attribute] to dynamic_attributes in case
        # that there are no dynamic features. Then one would get an error.

        _, df_target = feature_processor_new.extract_dfs(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            key_activities=key_activities,
            static_attributes=[],
            dynamic_attributes=[target_attribute_for_dyn],
            is_closed_indicator=is_closed_indicator,
            target_variable=target_variable,
            filters=self.filters,
        )
        start = timeit.default_timer()
        # Create the true targets
        self.df_target = self._create_true_target_df(df_target, target_attribute)

        # Get final df_x
        self.df_x = self._get_final_df_x(df_x, self.df_target)
        stop = timeit.default_timer()
        print("Time for creating the real dataframes: ", stop - start)
        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=used_static_attributes + used_dynamic_attributes,
            target_attributes=target_attribute,
            valid_target_values=None,
            invalid_target_replacement=None,
            min_counts_perc=self.min_attr_count_perc,
            max_counts_perc=self.max_attr_count_perc,
        )

        self.df_x, self.df_target, self.target_features, self.features = pp.process()

        statistics_computer = StatisticsComputer(
            features=self.features,
            target_features=self.target_features,
            df_x=self.df_x,
            df_target=self.df_target,
        )
        statistics_computer.compute_all_statistics()
        self.df_x, self.df_target = post_processing.remove_nan(
            df_x=self.df_x,
            df_target=self.df_target,
            features=self.features,
            target_features=self.target_features,
            th_remove_col=self.th_remove_col,
        )

    def _create_true_target_df(
        self, df_target: pd.DataFrame, target_attr: DecisionToActivityAttribute
    ) -> pd.DataFrame:
        """Create the true target df from the original df_target. It will just have
        the source activity as index and not the target activities anymore. Also,
        the column of the target_attr attribute will have the eventually following
        defined target activities or 'None' if none of the defined target activities
        eventually follow. If the source activity happens several times in a row
        without one of the target activities being between them, only the first
        occurrence of the source activity is held. The others are discarded. So,
        if S is the source activity and T is the target activity and A is another
        activity and the trace goes like this: S(1)->A->A-S(2)->S(3)->A->T, then the
        features are taken from S(1) and the decision goes to T. The other source
        activities are discarded."""
        target_col = target_attr.attribute_name
        # Add shifted columns of target column
        df_target_shifted_minus_1 = pd.DataFrame(
            index=df_target.index,
            columns=["shifted_-1"],
            data=df_target[target_col].shift(-1).values,
        )
        df_target_shifted_plus_1 = pd.DataFrame(
            index=df_target.index,
            columns=["shifted_+1"],
            data=df_target[target_col].shift(1).values,
        )
        df_with_shifts = pd.concat(
            [df_target, df_target_shifted_minus_1, df_target_shifted_plus_1], axis=1
        )

        # Create a new column 'shifted_-1_with_None'. Set it to the
        # values in 'shifted_-1' unless the last value in a group (A group of the
        # same index values). This way it is identified if a source activity doesn't
        # have an eventually following target activity in the same case.
        # First need to set level 0 index (case level) values to a column
        df_with_shifts["index"] = df_with_shifts.index.get_level_values(0)
        df_with_shifts["previous_index"] = df_with_shifts["index"].shift(1)
        df_with_shifts["next_index"] = df_with_shifts["index"].shift(-1)
        df_with_shifts["shifted_-1_with_None"] = df_with_shifts["shifted_-1"]
        df_with_shifts.loc[
            df_with_shifts["index"] != df_with_shifts["next_index"],
            "shifted_-1_with_None",
        ] = None

        # groups = df_with_shifts.groupby(
        #    (df_with_shifts["index"].shift() != df_with_shifts["index"]).cumsum()
        # )
        # for name, group in groups:
        #    idxs = group.index[-1]
        #    df_with_shifts.loc[idxs, "shifted_-1"] = None

        # If the source activity is not also a target activity:
        # Set shifted_-1 to the last value in a group (grouped by index and target
        # column). This is needed for the case that the source activity happens
        # multiple times in a row. For the first occurrence of the source activity in
        # such a group we still want to get the eventually following target activity.
        if self.source_activity not in self.target_activities:
            # Add column 'shifted_-1_not_same' with activity (target_col) if shifted_-1
            # is not the same as Activity. Else None
            df_with_shifts["shifted_-1_not_same"] = df_with_shifts[
                "shifted_-1_with_None"
            ]
            df_with_shifts.loc[
                df_with_shifts[target_col] == df_with_shifts["shifted_-1"],
                "shifted_-1_not_same",
            ] = None

            # Now bfill the None values to get the next activity in target_activities
            # df_with_shifts['shifted_-1_not_same'] = df_with_shifts[
            #    'shifted_-1_not_same'].fillna(method='bfill')
            # Now bfill in groups of indexes to get the next valid target activity
            df_with_shifts["shifted_-1_with_None"] = df_with_shifts.groupby("index")[
                "shifted_-1_not_same"
            ].bfill()
            self.df_with_shifts = df_with_shifts
            """
            groups = df_with_shifts.groupby(
                (
                    (df_with_shifts["index"].shift() != df_with_shifts["index"])
                    | (df_with_shifts[target_col].shift() != df_with_shifts[target_col])
                ).cumsum()
            )
            for name, group in groups:
                if len(group) <= 1:
                    continue
                idxs = group.index
                df_with_shifts.loc[idxs, "shifted_-1"] = group["shifted_-1"].iloc[-1]
            """

        # Only keep the rows where target_col == source activity
        df_only_sources = df_with_shifts[
            (df_with_shifts[target_col] == self.source_activity)
        ]

        # Remove the rows where target_col==self.source_activity and
        # shifted_+1==self.source_activity. Gets only the first
        # occurrences in a group of the source activity.
        if self.source_activity not in self.target_activities:
            indices_to_drop_not_first = df_with_shifts[
                (
                    (df_with_shifts[target_col] == self.source_activity)
                    & (df_with_shifts["shifted_+1"] == self.source_activity)
                    & (df_with_shifts["previous_index"] == df_with_shifts["index"])
                )
            ].index
            df_target = df_only_sources[
                ~df_only_sources.index.isin(indices_to_drop_not_first)
            ]
            # df_target = df_only_sources.drop(index=indices_to_drop_not_first)
        else:
            df_target = df_only_sources

        # The real target values are in 'shifted_-1' now. Copy it to the target_col
        # column.
        df_target[target_col] = df_target["shifted_-1_with_None"]

        return pd.DataFrame(df_target[target_col])

    def _get_final_df_x(self, df_x: pd.DataFrame, df_target: pd.DataFrame):
        """Get the final df_x. This is done by removing the rows which indices are
        not in df_target."""
        df_x = df_x.loc[df_target.index, :]
        return df_x

    def _gen_static_attr_list(self, min_attr_count: int, max_attr_count: int):
        """Gen list of used static attributes."""
        static_attributes_list = []
        # Add attributes that always need to be added for processing
        static_attributes_list.append(
            static_attributes.StartActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )
        static_attributes_list.append(
            static_attributes.EndActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )

        self.case_duration_attribute = static_attributes.CaseDurationAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            time_aggregation=self.time_unit,
            is_feature=False,
            is_class_feature=False,
        )
        static_attributes_list.append(self.case_duration_attribute)
        # Add used attributes

        if (
            self.start_activity_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            static_attributes_list.append(
                static_attributes.StartActivityAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if (
            self.case_table_col_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            case_table_col_attirbutes = (
                feature_processor_new.gen_case_column_attributes_multi_table(
                    process_config=self.process_config,
                    table_column_dict=self.considered_case_level_table_cols,
                    is_feature=True,
                    is_class_feature=False,
                )
            )
            static_attributes_list = static_attributes_list + case_table_col_attirbutes

        return static_attributes_list

    def _gen_dynamic_attr_list(self, min_attr_count: int, max_attr_count: int):
        """Gen list of used static attributes."""
        dynamic_attributes_list = []

        # Add used attributes

        if (
            self.current_activity_col_attr_descriptor
            in self.used_dynamic_attribute_descriptors
        ):
            current_activity_col_attributes = (
                feature_processor_new.gen_current_activity_col_attributes(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    columns=self.considered_activity_table_cols,
                    is_feature=True,
                    is_class_feature=False,
                )
            )
            dynamic_attributes_list = (
                dynamic_attributes_list + current_activity_col_attributes
            )

        if (
            self.previous_activity_occurrence_attr_descriptor
            in self.used_dynamic_attribute_descriptors
        ):
            previous_activity_occ_attributes = (
                feature_processor_new.gen_dynamic_activity_occurence_attributes(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    activities=None,
                    min_vals=min_attr_count,
                    max_vals=max_attr_count,
                    is_feature=True,
                    is_class_feature=False,
                    filters=self.filters,
                )
            )
            dynamic_attributes_list = (
                dynamic_attributes_list + previous_activity_occ_attributes
            )

        return dynamic_attributes_list


class ReworkProcessor(UseCaseProcessor):
    """Feature Processor for the case duration use case."""

    # attributes that can be used for this use case
    work_in_progress_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.WorkInProgressAttribute,
        display_name=static_attributes.WorkInProgressAttribute.display_name,
        description=static_attributes.WorkInProgressAttribute.description,
    )

    wip_case_start_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.WorkInProgressCaseStartAttribute,
        display_name=static_attributes.WorkInProgressCaseStartAttribute.display_name,
        description=static_attributes.WorkInProgressCaseStartAttribute.description,
    )

    start_activity_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.StartActivityAttribute,
        display_name=static_attributes.StartActivityAttribute.display_name,
        description=static_attributes.StartActivityAttribute.description,
    )

    case_table_col_attr_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.CaseTableColumnAttribute,
        display_name=static_attributes.CaseTableColumnAttribute.display_name,
        description=static_attributes.CaseTableColumnAttribute.description,
    )

    potential_static_attributes_descriptors = [
        start_activity_attr_descriptor,
        work_in_progress_attr_descriptor,
        wip_case_start_attr_descriptor,
        case_table_col_attr_descriptor,
    ]
    potential_dynamic_attributes_descriptors = []
    # the target attribute that is used for this use case
    target_attribute_descriptor = AttributeDescriptor(
        attribute_type=static_attributes.ReworkOccurrenceAttribute,
        display_name=static_attributes.ReworkOccurrenceAttribute.display_name,
        description=static_attributes.ReworkOccurrenceAttribute.description,
    )

    def __init__(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        used_static_attribute_descriptors: List[AttributeDescriptor],
        used_dynamic_attribute_descriptors: List[AttributeDescriptor],
        considered_activity_table_cols: List[str],
        considered_case_level_table_cols: Dict[str, List[str]],
        is_closed_query: pql.PQLColumn,
        rework_activities: List[str],
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            process_config=process_config,
            used_static_attribute_descriptors=used_static_attribute_descriptors,
            used_dynamic_attribute_descriptors=used_dynamic_attribute_descriptors,
            target_attribute_descriptor=self.target_attribute_descriptor,
            considered_activity_table_cols=considered_activity_table_cols,
            considered_case_level_table_cols=considered_case_level_table_cols,
            is_closed_query=is_closed_query,
            **kwargs,
        )
        self.activity_table_str = activity_table_str
        self.rework_activities = rework_activities
        self.time_unit = time_unit
        self.start_date = start_date
        self.end_date = end_date

    def process(self):
        date_filters = feature_processor_new.date_filter_PQL(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.filters = self.filters + date_filters

        is_closed_indicator = feature_processor_new.all_cases_closed_query(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            name="IS_CLOSED",
        )

        self.num_cases = feature_processor_new.get_number_cases(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            filters=self.filters,
        )

        (
            min_attr_count,
            max_attr_count,
        ) = feature_processor_new.compute_min_max_attribute_counts_PQL(
            self.min_attr_count_perc,
            self.max_attr_count_perc,
            process_config=self.process_config,
            activity_table_name=self.activity_table_str,
            chunksize=self.chunksize,
            filters=self.filters,
        )

        used_static_attributes = self._gen_static_attr_list(
            min_attr_count=min_attr_count, max_attr_count=max_attr_count
        )

        self.df_timestamp_column = "Start activity Time"

        target_attributes = self._gen_target_attributes()
        # Define a default target variable
        target_variable = target_attributes[0].pql_query
        target_variable.name = "dummy"
        used_static_attributes += target_attributes
        # Get DataFrames
        df_x, df_target = feature_processor_new.extract_dfs(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            static_attributes=used_static_attributes,
            dynamic_attributes=[],
            is_closed_indicator=is_closed_indicator,
            target_variable=target_variable,
            filters=self.filters,
        )

        self.df_x, self.df_target = self._get_real_dfs(
            df_x=df_x, target_attributes=target_attributes
        )

        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=used_static_attributes,
            target_attributes=target_attributes,
            valid_target_values=None,
            invalid_target_replacement=None,
            min_counts_perc=self.min_attr_count_perc,
            max_counts_perc=self.max_attr_count_perc,
        )

        self.df_x, self.df_target, self.target_features, self.features = pp.process()

        statistics_computer = StatisticsComputer(
            features=self.features,
            target_features=self.target_features,
            df_x=self.df_x,
            df_target=self.df_target,
        )
        statistics_computer.compute_all_statistics()
        self.df_x, self.df_target = post_processing.remove_nan(
            df_x=self.df_x,
            df_target=self.df_target,
            features=self.features,
            target_features=self.target_features,
            th_remove_col=self.th_remove_col,
        )

    def _get_real_dfs(self, df_x, target_attributes):
        columns_target = [attr.attribute_name for attr in target_attributes]
        columns_target_extended = columns_target.copy()
        columns_target_extended.append("Start activity Time")
        columns_target_extended.append("End activity Time")
        df_target = df_x[columns_target].copy()
        df_x.drop(columns=columns_target, axis=1, inplace=True)
        return df_x, df_target

    def _gen_target_attributes(self):
        """Gen target attributes"""
        target_attributes = []
        for act in self.rework_activities:
            attr = static_attributes.ReworkOccurrenceAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
                activity=act,
                is_feature=False,
                is_class_feature=True,
            )
            target_attributes.append(attr)
        return target_attributes

    def _gen_static_attr_list(self, min_attr_count: int, max_attr_count: int):
        """Gen list of used static attributes."""
        static_attributes_list = []
        # Add attributes that always need to be added for processing
        static_attributes_list.append(
            static_attributes.StartActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )
        static_attributes_list.append(
            static_attributes.EndActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=self.activity_table_str,
            )
        )

        self.case_duration_attribute = static_attributes.CaseDurationAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            time_aggregation=self.time_unit,
            is_feature=False,
            is_class_feature=False,
        )
        static_attributes_list.append(self.case_duration_attribute)

        # Add used attributes
        if (
            self.work_in_progress_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            static_attributes_list.append(
                static_attributes.WorkInProgressAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    aggregation="AVG",
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if (
            self.start_activity_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            static_attributes_list.append(
                static_attributes.StartActivityAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if (
            self.wip_case_start_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            static_attributes_list.append(
                static_attributes.WorkInProgressCaseStartAttribute(
                    process_config=self.process_config,
                    activity_table_str=self.activity_table_str,
                    is_feature=True,
                    is_class_feature=False,
                )
            )

        if (
            self.case_table_col_attr_descriptor
            in self.used_static_attribute_descriptors
        ):
            case_table_col_attirbutes = (
                feature_processor_new.gen_case_column_attributes_multi_table(
                    process_config=self.process_config,
                    table_column_dict=self.considered_case_level_table_cols,
                    is_feature=True,
                    is_class_feature=False,
                )
            )
            static_attributes_list = static_attributes_list + case_table_col_attirbutes

        return static_attributes_list
