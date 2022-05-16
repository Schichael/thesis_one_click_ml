from typing import Any, Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from prediction_builder.data_extraction import PQLExtractor
from prediction_builder.data_extraction import ProcessModelFactory
from prediction_builder.data_extraction import StaticPQLExtractor
from pycelonis.celonis_api.event_collection import data_model
from pycelonis.celonis_api.pql.pql import PQL
from pycelonis.celonis_api.pql.pql import PQLColumn
from pycelonis.celonis_api.pql.pql import PQLFilter

from one_click_analysis import utils
from one_click_analysis.feature_processing.attributes.attribute import AttributeDataType
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    ActivityDurationAttribute,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    CurrentActivityColumnAttribute,
)

from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    DynamicAttribute,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    NextActivityAttribute,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    PreviousActivityOccurrenceAttribute,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    PreviousActivityColumnAttribute,
)

from one_click_analysis.feature_processing.attributes.static_attributes import (
    ActivityOccurenceAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    CaseDurationAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    CaseTableColumnAttribute,
)

from one_click_analysis.feature_processing.attributes.static_attributes import (
    EndActivityAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    EndActivityTimeAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    NumericActivityTableColumnAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    StartActivityAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    StartActivityTimeAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    StaticAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    WorkInProgressAttribute,
)
from one_click_analysis.feature_processing.post_processing import PostProcessor
from one_click_analysis.feature_processing.statistics_computer import StatisticsComputer
from one_click_analysis.feature_processing.util_queries import extract_transitions
from one_click_analysis.process_config.process_config import (
    ProcessConfig,
    ActivityTable,
)

pd.options.mode.chained_assignment = None


class FeatureProcessor:
    """
    The FeatureProcessor fetches data from the Celonis database and generates the
    desired features
    """

    def __init__(
        self,
        process_config: ProcessConfig,
        chunksize: int = 10000,
        min_attr_count_perc: float = 0.02,
        max_attr_count_perc: float = 0.98,
    ):

        self.process_config = process_config
        self.dm = self.process_config.dm
        self.chunksize = chunksize
        self.min_attr_count_perc = min_attr_count_perc
        self.max_attr_count_perc = max_attr_count_perc

        self.categorical_types = ["STRING", "BOOLEAN"]
        self.numerical_types = ["INTEGER", "FLOAT"]

        self.activity_col = None
        self.eventtime_col = None
        self.sort_col = None
        self.eventtime_col = None

        self.label_col = None
        self.activity_start = "Activity_START"
        self.activity_end = "Activity_END"
        self.config_file_name = None
        self.attributes = []
        self.static_attributes = []
        self.dynamic_attributes = []
        self.attributes_dict = {}
        self.labels = []
        self.labels_dict = {}
        self.df_timestamp_column = ""
        self.filters = []  # general PQL filters from configuration
        (
            self.min_attr_count,
            self.max_attr_count,
        ) = self.compute_min_max_attribute_counts_PQL(
            min_attr_count_perc, max_attr_count_perc
        )
        self.df = None
        self.minor_attrs = []

        # HERE ARE THE NEW VARIABLES SINCE THE BIG REFACTORING
        self.features = []
        self.target_features = []
        self.df_x = None
        self.df_target = None

    def get_df_with_filters(self, query: PQL):
        for f in self.filters:
            query.add(f)
        return self.dm.get_data_frame(query, chunksize=self.chunksize)

    def get_valid_vals(
        self,
        table_name: str,
        column_names: List[str],
        process_config: ProcessConfig,
        min_vals: int = 0,
        max_vals: int = np.inf,
    ):
        """Get values of columns that occur in enough cases. This is done with PQL
        and shall be used before the actual column is queried for the resulting
        DataFrame.

        :param table_name:
        :param process_config: the ProcessConfig object
        :param case_table_str: the name of the associated case table
        :param column_names:
        :param min_vals:
        :param max_vals:
        :return: dictionary with keys=column names and values: List of the valid
        values in the column.
        """
        table = process_config.table_dict[table_name]
        # if the table is an activity table, we need its case table for counting (
        # Actually not really necessary since we can also achieve the count with
        # another query that only uses the activity table (COUNT(DISTINCT(
        # "activity_table"."column_name")) ). But idk if it's better to do it with
        # the case table)
        if isinstance(table, ActivityTable):
            target_table_str = table.case_table_str
        else:
            target_table_str = table_name

        query = PQL()
        for col_name in column_names:
            query.add(
                PQLColumn(
                    name=col_name + "_values",
                    query='DISTINCT("' + table_name + '"."' + col_name + '")',
                )
            )
            query.add(
                PQLColumn(
                    name=col_name + "_count",
                    query='COUNT_TABLE("' + target_table_str + '")',
                )
            )
            # Can just query enough occurences using filter

        df_val_counts = self.get_df_with_filters(query)
        cols = df_val_counts.columns.tolist()

        valid_vals_dict = {}
        for col in cols:
            if not col.endswith("_values"):
                continue
            col_name = col[:-7]

            count_col_name = col_name + "_count"
            valid_vals_dict[col_name] = df_val_counts[
                (df_val_counts[count_col_name] >= min_vals)
                & (df_val_counts[count_col_name] <= max_vals)
            ][col].values.tolist()
        return valid_vals_dict

    def date_filter_PQL(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Add date filter to self.filters
        :param process_config: ProcessConfig object
        :param activity_table_str: activity table name
        :param start_date: date in the form yyyy-mm-dd
        :param end_date: date in the form yyyy-mm-dd
        :return:
        """
        activity_table = process_config.table_dict[activity_table_str]
        case_table_str = activity_table.case_table_str
        if start_date:
            date_str_pql = f"{{d'{start_date}'}}"
            filter_str = (
                f'PU_FIRST("{case_table_str}", '
                f'"{activity_table.table_str}"."'
                f'{activity_table.eventtime_col_str}") >= {date_str_pql}'
            )
            filter_start = PQLFilter(filter_str)
            self.filters.append(filter_start)

        if end_date:
            date_str_pql = f"{{d'{end_date}'}}"
            filter_str = (
                f'PU_FIRST("{case_table_str}", '
                f'"{activity_table.table_str}"."'
                f'{activity_table.eventtime_col_str}") <= {date_str_pql}'
            )
            filter_start = PQLFilter(filter_str)
            self.filters.append(filter_start)

    def _gen_dynamic_activity_occurence_attributes(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        activities: List[str] = None,
        min_vals: int = 0,
        max_vals: int = np.inf,
        is_feature: bool = True,
        is_class_feature: bool = False,
    ) -> List[PreviousActivityOccurrenceAttribute]:
        """Generates the static ActivitiyOccurenceAttributes. If no activities are
        given, all activities are used and it's checked for min and max values. If
        activities are given, it is not checked for min and max occurences."""
        activity_table = process_config.table_dict[activity_table_str]
        if activities is None:
            activities = self.get_valid_vals(
                table_name=activity_table.table_str,
                column_names=[activity_table.activity_col_str],
                process_config=process_config,
                min_vals=min_vals,
                max_vals=max_vals,
            )[activity_table.activity_colu_str]

        activity_occ_attributes = []
        for activity in activities:
            attr = PreviousActivityOccurrenceAttribute(
                process_config=process_config,
                activity_table_str=activity_table_str,
                activity=activity,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            activity_occ_attributes.append(attr)

        return activity_occ_attributes

    def _gen_static_activity_occurence_attributes(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        activities: List[str] = None,
        min_vals: int = 0,
        max_vals: int = np.inf,
        is_feature: bool = True,
        is_class_feature: bool = False,
    ) -> List[ActivityOccurenceAttribute]:
        """Generates the static ActivitiyOccurenceAttributes. If no activities are
        given, all activities are used and it's checked for min and max values. If
        activities are given, it is not checked for min and max occurences."""

        if activities is None:
            activity_table = process_config.table_dict[activity_table_str]

            activities = self.get_valid_vals(
                table_name=activity_table_str,
                column_names=[activity_table.activity_col_str],
                process_config=process_config,
                min_vals=min_vals,
                max_vals=max_vals,
            )[activity_table.activity_col_str]

        activity_occ_attributes = []
        for activity in activities:
            attr = ActivityOccurenceAttribute(
                process_config=process_config,
                activity_table_str=activity_table_str,
                activity=activity,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            activity_occ_attributes.append(attr)

        return activity_occ_attributes

    def _gen_static_numeric_activity_table_attributes(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        columns: List[str],
        aggregation: str,
        is_feature: bool = True,
        is_class_feature: bool = False,
    ):
        """Generate list of static NumericActivityTableColumnAttributes

        :param columns:
        :param aggregation:
        :param is_feature:
        :param is_class_feature:
        :return:
        """
        attrs = []
        for column in columns:
            attr = NumericActivityTableColumnAttribute(
                process_config=process_config,
                activity_table_str=activity_table_str,
                column_name=column,
                aggregation=aggregation,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            attrs.append(attr)
        return attrs

    def _gen_case_column_attributes_single_table(self, process_config: ProcessConfig,
                                          table_str: str, columns: List[str],
                                          attribute_datatype: AttributeDataType,
                                          is_feature: bool = True,
                                          is_class_feature: bool = False,
                                          ):
        """Generate list of static CaseTableColumnAttributes for one table

        :param process_config:
        :param table_str:
        :param columns:
        :param attribute_datatype:
        :param is_feature:
        :param is_class_feature:
        :return:
        """
        attrs = []
        for column in columns:
            attr = CaseTableColumnAttribute(process_config=process_config,
                table_name=table_str, attribute_datatype=attribute_datatype,
                column_name=column, is_feature=is_feature,
                is_class_feature=is_class_feature, )
            attrs.append(attr)
        return attrs

    def _gen_numeric_case_column_attributes_single_table(
        self,
        process_config: ProcessConfig,
        table_str: str,
        columns: List[str],
        is_feature: bool = True,
        is_class_feature: bool = False,
    ):
        """Convenience method. Generate list of static numeric CaseTableColumnAttributes

        :param columns:
        :param is_feature:
        :param is_class_feature:
        :return:
        """
        return self._gen_case_column_attributes_single_table(
            process_config=process_config,table_str=table_str,columns=columns,
            attribute_datatype=AttributeDataType.NUMERICAL, is_feature=is_feature,
            is_class_feature=is_class_feature
        )

    def _gen_categorical_case_column_attributes_single_table(
        self,
        process_config: ProcessConfig,
        table_str: str,
        columns: List[str],
        is_feature: bool = True,
        is_class_feature: bool = False,
    ):
        """Convenience method. Generate list of static
        categorical CaseTableColumnAttributes

        :param columns:
        :param is_feature:
        :param is_class_feature:
        :return:
        """
        return self._gen_case_column_attributes_single_table(
            process_config=process_config,table_str=table_str,columns=columns,
            attribute_datatype=AttributeDataType.CATEGORICAL, is_feature=is_feature,
            is_class_feature=is_class_feature
        )

    def _gen_case_column_attributes_multi_table(
        self,
        process_config: ProcessConfig,
        table_column_dict: Dict[str, List[str]],
        is_feature: bool = True,
        is_class_feature: bool = False,
    ):
        """Generate list of static CaseTableColumnAttributes for several tables and
        for numeric and categorical columns.

        :param process_config:
        :param table_column_dict: dictionary with table names and columns that shall
        be considered.
        :param is_feature:
        :param is_class_feature:
        :return:
        """
        attributes = []
        for table_name, all_columns in table_column_dict.items():
            # Get all categorical and numerical columns from table
            cat_cols, num_cols = \
                self.process_config.get_categorical_numerical_columns(table_name)
            # Now only use those columns that are defined in the table_column_dict
            cat_cols = utils.list_intersection(cat_cols, all_columns)
            num_cols = utils.list_intersection(num_cols, all_columns)
            # Get attributes for categorical columns
            num_attributes = self._gen_numeric_case_column_attributes_single_table(
                process_config=process_config,
                columns=num_cols,
            is_feature=is_feature,
            is_class_feature=is_class_feature,
            table_str=table_name,
            )
            cat_attributes = self._gen_categorical_case_column_attributes_single_table(
                process_config=process_config, columns=cat_cols, is_feature=is_feature,
                is_class_feature=is_class_feature, table_str=table_name,)
            attributes = attributes + num_attributes + cat_attributes

        return attributes




    def _default_target_query(
        self, process_config: ProcessConfig, activity_table_str: str, name="TARGET"
    ):
        """Return PQLColumn that evaluates to 1 for all cases wth default column name
        TARGET"""
        activity_table = process_config.table_dict[activity_table_str]
        q_str = (
            f'CASE WHEN PU_COUNT("{activity_table.case_table_str}", '
            f'"{activity_table.table_str}"."'
            f'{activity_table.activity_col_str}") >= 1 THEN 1 ELSE 0 END'
        )
        return PQLColumn(name=name, query=q_str)

    def _all_cases_closed_query(
        self, process_config: ProcessConfig, activity_table_str: str, name="IS_CLOSED"
    ):
        """Return PQLColumn that evaluates to 1 for all cases with default column name
        IS_CLOSED"""
        activity_table = process_config.table_dict[activity_table_str]
        q_str = (
            f'CASE WHEN PU_COUNT("{activity_table.case_table_str}", '
            f'"{activity_table.table_str}"."'
            f'{activity_table.activity_col_str}") >= 1 THEN 1 ELSE 0 END'
        )
        return PQLColumn(name=name, query=q_str)

    def extract_dfs(
        self,
        process_config: ProcessConfig,
        activity_table_str: str,
        static_attributes: List[StaticAttribute],
        dynamic_attributes: List[DynamicAttribute],
        is_closed_indicator: PQLColumn,
        target_variable: PQLColumn,
    ) -> Tuple[Any, ...]:
        activity_table = process_config.table_dict[activity_table_str]
        process_model = activity_table.process_model
        # Get PQLColumns from the attributes
        process_model.global_filters = self.filters
        static_attributes_pql = [attr.pql_query for attr in static_attributes]
        if dynamic_attributes:
            dynamic_attributes_pql = [attr.pql_query for attr in dynamic_attributes]
            extractor = PQLExtractor(
                process_model=process_model,
                target_variable=target_variable,
                is_closed_indicator=is_closed_indicator,
                static_features=static_attributes_pql,
                dynamic_features=dynamic_attributes_pql,
            )
            df_x, df_y = extractor.get_closed_cases(self.dm, only_last_state=False)
        else:
            extractor = StaticPQLExtractor(
                process_model=process_model,
                target_variable=target_variable,
                is_closed_indicator=is_closed_indicator,
                static_features=static_attributes_pql,
            )
            df_x, df_y = extractor.get_closed_cases(self.dm, only_last_state=False)
        # df_y is a Series object. Make it a DataFrame
        df_y = pd.DataFrame(df_y)
        return df_x, df_y

    def run_total_time_PQL(
        self,
        activity_table_str: str,
        considered_activity_table_cols: List[str],
        considered_case_level_table_cols: Dict[str, List[str]],
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Run feature processing for total case time analysis.

        :param time_unit: time unit to use. E.g. DAYS if attributes shall be in days.
        :return:
        """
        # Add filters for start and end date
        self.date_filter_PQL(self.process_config, activity_table_str, start_date,
                             end_date)
        # Define closed case (currently define that every case is a closed case)
        is_closed_indicator = self._all_cases_closed_query()

        # self.process_model.global_filters = self.filters

        # self.transitions_df = extract_transitions(
        #    process_model=self.process_model,
        #    dm=self.dm,
        #    is_closed_indicator=is_closed_indicator,
        # )

        static_attributes = [
            WorkInProgressAttribute(
                process_config=self.process_config,
                activity_table_str=activity_table_str,
                aggregation="AVG",
                is_feature=True,
                is_class_feature=False,
            ),
            StartActivityAttribute(
                process_config=self.process_config,
                activity_table_str=activity_table_str,
                is_feature=True,
                is_class_feature=False,
            ),
            EndActivityAttribute(
                process_config=self.process_config,
                activity_table_str=activity_table_str,
                is_feature=True,
                is_class_feature=False,
            ),
            StartActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=activity_table_str,
            ),
            EndActivityTimeAttribute(
                process_config=self.process_config,
                activity_table_str=activity_table_str,
            )
            # ActivityOccurenceAttribute(process_model=self.process_model,
            #    aggregation="AVG", is_feature=True, is_class_feature=False,
            # )
        ]

        # First see which activities happen often enough to be used. Then create the
        # attributes for those.
        activity_occ_attributes = self._gen_static_activity_occurence_attributes(
            process_config=self.process_config,
            activity_table_str=activity_table_str,
            is_feature=True,
            min_vals=self.min_attr_count,
            max_vals=self.max_attr_count,
        )
        _, num_actiivty_table_cols = \
            self.process_config.get_categorical_numerical_columns(activity_table_str)

        # Use just the activity table columns that are in considered_activity_table_cols
        #cat_actiivty_table_cols = utils.list_intersection(cat_actiivty_table_cols,
        # considered_activity_table_cols)
        num_actiivty_table_cols = utils.list_intersection(num_actiivty_table_cols, considered_activity_table_cols)

        numeric_activity_column_attributes = (
            self._gen_static_numeric_activity_table_attributes(
                process_config=self.process_config, activity_table_str=activity_table_str,
                columns=num_actiivty_table_cols,
                aggregation="AVG",
                is_feature=True,
                is_class_feature=False,
            )
        )
        case_table_column_attrs = self._gen_case_column_attributes_multi_table(
            process_config=self.process_config,
            table_column_dict=considered_case_level_table_cols, is_feature=True,
            is_class_feature=False
        )
        """
        categorical_case_table_column_attrs = (
            self._gen_categorical_case_column_attributes_single_table(
                columns=self.static_categorical_cols,
                is_feature=True,
                is_class_feature=False,
            )
        )
        numeric_case_table_column_attrs = self._gen_numeric_case_column_attributes(
            columns=self.static_numerical_cols,
            is_feature=True,
            is_class_feature=False,
        )
        """

        static_attributes = (
            static_attributes
            + activity_occ_attributes
            + numeric_activity_column_attributes
            + case_table_column_attrs
        )

        self.static_attributes = static_attributes

        dynamic_attributes = []

        self.dynamic_attributes = dynamic_attributes

        self.df_timestamp_column = "Start activity Time"

        target_attribute = CaseDurationAttribute(
            process_config=self.process_config,activity_table_str=activity_table_str,
            time_aggregation=time_unit,
            is_feature=False,
            is_class_feature=True,
        )

        # Define a default target variable
        target_variable = target_attribute.pql_query

        # Get DataFrames
        self.df_x, self.df_target = self.extract_dfs(
            process_config=self.process_config,
            activity_table_str=activity_table_str,
            static_attributes=static_attributes,
            dynamic_attributes=dynamic_attributes,
            is_closed_indicator=is_closed_indicator,
            target_variable=target_variable,
        )

        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=static_attributes + dynamic_attributes,
            target_attribute=target_attribute,
            valid_target_values=None,
            invalid_target_replacement=None,
            min_counts_perc=0.02,
            max_counts_perc=0.98,
        )

        self.df_x, self.df_target, self.target_features, self.features = pp.process()

        statistics_computer = StatisticsComputer(
            features=self.features,
            target_features=self.target_features,
            df_x=self.df_x,
            df_target=self.df_target,
        )
        statistics_computer.compute_all_statistics()

        # return df_x, df_y, target_features, features

    def filter_transition_PQL(self, start_activity: str):
        # term_end_activities = (
        #    "(" + ", ".join("'" + act + "'" for act in end_activities) + ")"
        # )
        filter_str = f"PROCESS EQUALS '{start_activity}'"
        return PQLFilter(filter_str)

    def _get_previous_numerical_activity_col_attributes(self):
        attrs = []
        for col in self.dynamic_numerical_cols:
            attr = PreviousNumericalActivityColumnAttribute(
                process_model=self.process_model, column_name=col, is_feature=True
            )
            attrs.append(attr)
        return attrs

    def _get_current_numerical_activity_col_attributes(self):
        attrs = []
        for col in self.dynamic_numerical_cols:
            attr = CurrentNumericalActivityColumnAttribute(
                process_model=self.process_model, column_name=col, is_feature=True
            )
            attrs.append(attr)
        return attrs

    def _get_previous_categorical_activity_col_attributes(self):
        attrs = []
        for col in self.dynamic_categorical_cols:
            attr = PreviousCategoricalActivityColumnAttribute(
                process_model=self.process_model, column_name=col, is_feature=True
            )
            attrs.append(attr)
        return attrs

    def _get_current_categorical_activity_col_attributes(self):
        attrs = []
        for col in self.dynamic_categorical_cols:
            attr = CurrentCategoricalActivityColumnAttribute(
                process_model=self.process_model, column_name=col, is_feature=True
            )
            attrs.append(attr)
        return attrs

    def run_decision_point_new(
        self,
        source_activity: str,
        target_activities: List[str],
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        invalid_target_replacement: str = "OTHER",
    ):
        """Run feature processing for total case time analysis.

        :param time_unit: time unit to use. E.g. DAYS if attributes shall be in days.
        :return:
        """
        # Add filters for start and end date
        self.date_filter_PQL(start_date, end_date)

        static_attributes = [
            StartActivityAttribute(
                process_model=self.process_model,
                is_feature=True,
                is_class_feature=False,
            ),
            StartActivityTimeAttribute(process_model=self.process_model),
            EndActivityTimeAttribute(process_model=self.process_model),
            CaseDurationAttribute(
                process_model=self.process_model,
                time_aggregation=time_unit,
                is_feature=False,
                is_class_feature=False,
            ),
            # ActivityOccurenceAttribute(process_model=self.process_model,
            #   aggregation="AVG", is_feature=True, is_class_feature=False,
            # )
        ]

        # First see which activities happen often enough to be used. Then create the
        # attributes for those.

        categorical_case_table_column_attrs = (
            self._gen_categorical_case_column_attributes_single_table(
                columns=self.static_categorical_cols,
                is_feature=True,
                is_class_feature=False,
            )
        )
        numeric_case_table_column_attrs = self._gen_numeric_case_column_attributes(
            columns=self.static_numerical_cols,
            is_feature=True,
            is_class_feature=False,
        )

        static_attributes = (
            static_attributes
            + categorical_case_table_column_attrs
            + numeric_case_table_column_attrs
        )

        self.static_attributes = static_attributes

        dynamic_attributes = []
        num_act_cols = self._get_current_numerical_activity_col_attributes()
        cat_act_cols = self._get_current_categorical_activity_col_attributes()
        activity_occ_attributes = self._gen_dynamic_activity_occurence_attributes(
            is_feature=True, min_vals=self.min_attr_count, max_vals=self.max_attr_count
        )

        dynamic_attributes = (
            dynamic_attributes + num_act_cols + cat_act_cols + activity_occ_attributes
        )

        self.dynamic_attributes = dynamic_attributes

        self.df_timestamp_column = "Start activity Time"

        # Set key activity
        self.process_model.key_activities = [source_activity]

        target_attribute = NextActivityAttribute(
            process_model=self.process_model,
            is_class_feature=True,
            attribute_name="Transition to activity",
        )

        # Define closed case (currently define that every case is a closed case)
        is_closed_indicator = self._all_cases_closed_query()
        # Define a default target variable
        target_variable = target_attribute.pql_query

        # Get DataFrames
        self.df_x, self.df_target = self.extract_dfs(
            static_attributes=static_attributes,
            dynamic_attributes=dynamic_attributes,
            is_closed_indicator=is_closed_indicator,
            target_variable=target_variable,
        )

        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=static_attributes + dynamic_attributes,
            target_attribute=target_attribute,
            valid_target_values=target_activities,
            invalid_target_replacement=invalid_target_replacement,
            min_counts_perc=0.02,
            max_counts_perc=0.98,
        )

        self.df_x, self.df_target, self.target_features, self.features = pp.process()

        statistics_computer = StatisticsComputer(
            features=self.features,
            target_features=self.target_features,
            df_x=self.df_x,
            df_target=self.df_target,
        )
        statistics_computer.compute_all_statistics()

        # return df_x, df_y, target_features, features

    def _filter_next_activity(self, next_activity: str):
        pql_str = (
            f'ACTIVITY_LEAD("{self.process_model.activity_table_str}"."'
            f"{self.process_model.activity_column_str}\", 1) = '{next_activity}'"
        )

        filter_next_activity = PQLFilter(pql_str)
        self.filters.append(filter_next_activity)

    def _filter_prev_activity(self, prev_activity: str):
        pql_str = (
            f'ACTIVITY_LAG("{self.process_model.activity_table_str}"."'
            f"{self.process_model.activity_column_str}\", 1) = '{prev_activity}'"
        )

        filter_next_activity = PQLFilter(pql_str)
        self.filters.append(filter_next_activity)

    def run_transition_time(
        self,
        source_activity: str,
        target_activity: str,
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Run feature processing for total case time analysis.

        :param time_unit: time unit to use. E.g. DAYS if attributes shall be in days.
        :return:
        """
        # Add filters for start and end date
        self.date_filter_PQL(start_date, end_date)
        self._filter_prev_activity(prev_activity=source_activity)

        static_attributes = [
            StartActivityAttribute(
                process_model=self.process_model,
                is_feature=True,
                is_class_feature=False,
            ),
            StartActivityTimeAttribute(process_model=self.process_model),
            EndActivityTimeAttribute(process_model=self.process_model),
            # ActivityOccurenceAttribute(process_model=self.process_model,
            #    aggregation="AVG", is_feature=True, is_class_feature=False,
            # )
        ]

        # First see which activities happen often enough to be used. Then create the
        # attributes for those.

        categorical_case_table_column_attrs = (
            self._gen_categorical_case_column_attributes_single_table(
                columns=self.static_categorical_cols,
                is_feature=True,
                is_class_feature=False,
            )
        )
        numeric_case_table_column_attrs = self._gen_numeric_case_column_attributes(
            columns=self.static_numerical_cols,
            is_feature=True,
            is_class_feature=False,
        )

        static_attributes = (
            static_attributes
            + categorical_case_table_column_attrs
            + numeric_case_table_column_attrs
        )

        self.static_attributes = static_attributes

        dynamic_attributes = []
        current_num_act_cols = self._get_current_numerical_activity_col_attributes()
        current_cat_act_cols = self._get_current_categorical_activity_col_attributes()
        prev_num_act_cols = self._get_previous_numerical_activity_col_attributes()
        prev_cat_act_cols = self._get_previous_categorical_activity_col_attributes()

        dynamic_attributes = (
            dynamic_attributes
            + current_num_act_cols
            + current_cat_act_cols
            + prev_num_act_cols
            + prev_cat_act_cols
        )

        self.dynamic_attributes = dynamic_attributes

        self.df_timestamp_column = "Start activity Time"

        # Set key activity
        self.process_model.key_activities = [target_activity]

        target_attribute = ActivityDurationAttribute(
            process_model=self.process_model,
            unit=time_unit,
            is_feature=False,
            is_class_feature=True,
        )

        # Define closed case (currently define that every case is a closed case)
        is_closed_indicator = self._all_cases_closed_query()
        # Define a default target variable
        target_variable = target_attribute.pql_query

        # Get DataFrames
        self.df_x, self.df_target = self.extract_dfs(
            static_attributes=static_attributes,
            dynamic_attributes=dynamic_attributes,
            is_closed_indicator=is_closed_indicator,
            target_variable=target_variable,
        )
        # Round target values
        """
        self.df_target[target_attribute.attribute_name] = self.df_target[
            target_attribute.attribute_name
        ].round()
        self.df_target[target_attribute.attribute_name] = self.df_target[
            target_attribute.attribute_name
        ].astype(int)
        """

        pp = PostProcessor(
            df_x=self.df_x,
            df_target=self.df_target,
            attributes=static_attributes + dynamic_attributes,
            target_attribute=target_attribute,
            valid_target_values=None,
            invalid_target_replacement=None,
            min_counts_perc=0.02,
            max_counts_perc=0.98,
        )

        self.df_x, self.df_target, self.target_features, self.features = pp.process()

        statistics_computer = StatisticsComputer(
            features=self.features,
            target_features=self.target_features,
            df_x=self.df_x,
            df_target=self.df_target,
        )
        statistics_computer.compute_all_statistics()

        # return df_x, df_y, target_features, features

    def get_activities(self) -> pd.DataFrame:
        """Get DataFrame with the activities

        :return:
        """
        query_activities = (
            f'DISTINCT("{self.activity_table_name}".' f'"{self.activity_col}")'
        )
        pql_query = PQL()
        pql_query.add(PQLColumn(query=query_activities, name="activity"))
        df = self.dm.get_data_frame(pql_query)
        return df

    def get_activity_case_counts(self) -> pd.DataFrame:
        """Get DataFrame with the activities and the number of cases in which they
        occur.

        :return:
        """
        query_activities = f'"{self.activity_table_name}"."{self.activity_col}"'
        query_cases = f'COUNT_TABLE("{self.case_table_name}")'
        pql_query = PQL()
        pql_query.add(PQLColumn(query_activities, name="activity"))
        pql_query.add(PQLColumn(query_cases, name="case count"))
        df = self.dm.get_data_frame(pql_query)
        return df

    def set_latest_date_PQL(self):
        query = PQL()
        query += PQLColumn(
            "MAX("
            + '"'
            + self.activity_table_name
            + '"'
            + "."
            + '"'
            + self.eventtime_col
            + '"'
            + ")",
            "latest_date",
        )
        df = self.dm.get_data_frame(query, chunksize=self.chunksize)
        self.latest_date = df["latest_date"][0]

    def _conv_dtypes_PQL(
        self, df: pd.DataFrame, src_dtypes: List[str], target_dtype: str
    ) -> pd.DataFrame:
        """Convert columns of types src_dtypes to datatype target_dtype

        :param df: input DataFrame
        :param src_dtypes: list of data types to convert
        :param target_dtype: datatype to convert to
        :return: DatFrame with changed dtypes
        """
        df = df.copy()
        df[df.select_dtypes(src_dtypes).columns] = df.select_dtypes(src_dtypes).apply(
            lambda x: x.astype(target_dtype)
        )
        return df

    def compute_min_max_attribute_counts_PQL(
        self, min_counts_perc: float, max_counts_perc: float
    ) -> Tuple[int, int]:
        """Compute the minimum and maximum required attribute counts.

        :param min_counts_perc: minimum count percentage
        :param max_counts_perc: maximum count percentage
        :return: minimum and maximum attribute counts
        """
        query_num_cases = (
            'COUNT(DISTINCT "'
            + self.activity_table_name
            + '"."'
            + self.activity_case_key
            + '")'
        )
        pql_num_cases = PQL()
        pql_num_cases.add(PQLColumn(query_num_cases, "number cases"))
        df_num_cases = self.dm.get_data_frame(pql_num_cases, chunksize=self.chunksize)
        num_cases = df_num_cases["number cases"].values[0]
        min_attr_counts = round(num_cases * min_counts_perc)
        max_attr_counts = round(num_cases * max_counts_perc)

        return min_attr_counts, max_attr_counts
