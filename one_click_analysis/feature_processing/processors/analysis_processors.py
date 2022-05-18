import abc
from typing import Dict
from typing import List
from typing import Optional

from one_click_analysis.feature_processing import feature_processor_new
from one_click_analysis.feature_processing.attributes import static_attributes
from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDescriptor,
)
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
        **kwargs,
    ):
        self.process_config = process_config
        self.used_static_attribute_descriptors = used_static_attribute_descriptors
        self.used_dynamic_attribute_descriptors = used_dynamic_attribute_descriptors
        self.target_attribute_descriptor = target_attribute_descriptor
        self.considered_activity_table_cols = considered_activity_table_cols
        self.considered_case_level_table_cols = considered_case_level_table_cols
        self.chunksize = kwargs.get("chunksize", 10000)
        self.min_attr_count_perc = kwargs.get("min_attr_count_perc", 0.02)
        self.max_attr_count_perc = kwargs.get("max_attr_count_perc", 0.98)

        self.df_x = None
        self.df_target = None
        self.target_features = None
        self.features = None
        self.filters = []


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
        attribute_type=static_attributes.NumericActivityTableColumnAttribute,
        display_name=static_attributes.NumericActivityTableColumnAttribute.display_name,
        description=static_attributes.NumericActivityTableColumnAttribute.description,
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
        self.filters.append(date_filters)

        is_closed_indicator = feature_processor_new.all_cases_closed_query(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            name="IS_CLOSED",
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
            is_closed_indicator=is_closed_indicator,
            target_variable=target_variable,
            filters=self.filters,
        )

        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=used_static_attributes,
            target_attribute=target_attribute,
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

    def _gen_static_attr_list(self, min_attr_count: int, max_attr_count: int):
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
            static_attributes_list.append(activity_occ_attributes)

        return static_attributes_list
