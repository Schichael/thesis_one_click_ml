from typing import Any
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

from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    ActivityDurationAttribute,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    CurrentCategoricalActivityColumnAttribute,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    CurrentNumericalActivityColumnAttribute,
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
    PreviousCategoricalActivityColumnAttribute,
)
from one_click_analysis.feature_processing.attributes.dynamic_attributes import (
    PreviousNumericalActivityColumnAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    ActivityOccurenceAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    CaseDurationAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    CaseTableColumnCategoricalAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    CaseTableColumnNumericAttribute,
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

pd.options.mode.chained_assignment = None


class EmptyTable:
    """
    Helper class for the case that the datamodel has no case table.
    """

    def __init__(self):
        self.columns = []

    def __bool__(self):
        return False


class FeatureProcessor:
    """
    The FeatureProcessor fetches data from the Celonis database and generates the
    desired features
    """

    def __init__(
        self,
        dm: data_model,
        chunksize: int = 10000,
        min_attr_count_perc: float = 0.02,
        max_attr_count_perc: float = 0.98,
    ):

        self.dm = dm
        self.chunksize = chunksize
        self.min_attr_count_perc = min_attr_count_perc
        self.max_attr_count_perc = max_attr_count_perc

        self.categorical_types = ["STRING", "BOOLEAN"]
        self.numerical_types = ["INTEGER", "FLOAT"]

        self.activity_table_name = None
        self.case_table_name = None
        self.activity_table = None
        self.case_table = None
        self.case_case_key = None
        self.activity_case_key = None
        self.prefixes_case_key = "caseid_prefixes"
        self.activity_col = None
        self.eventtime_col = None
        self.sort_col = None
        self.eventtime_col = None
        self.process_model = None
        self.static_numerical_cols = []
        self.static_categorical_cols = []
        self.dynamic_numerical_cols = []
        self.dynamic_categorical_cols = []
        self.static_categorical_values = {}
        self.dynamic_categorical_values = {}
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
        self._init_datamodel(self.dm)
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

    def reset_fp(self):
        """Reset the feature_processor to its initial values."""
        self.__init__(
            self.dm, self.chunksize, self.min_attr_count_perc, self.max_attr_count_perc
        )

    def _init_datamodel(self, dm):
        """Initialize datamodel parameters needed for fetching data from the Celonis
        database

        :param dm: input Datamodel
        :return:
        """
        # get activity and case table IDs
        activity_table_id = dm.data["processConfigurations"][0]["activityTableId"]
        case_table_id = dm.data["processConfigurations"][0]["caseTableId"]
        self.activity_table = dm.tables.find(activity_table_id)
        self.eventtime_col = dm.data["processConfigurations"][0]["timestampColumn"]
        self.sort_col = dm.data["processConfigurations"][0]["sortingColumn"]
        self.activity_col = dm.data["processConfigurations"][0]["activityColumn"]
        self.activity_table_name = self.activity_table.name

        if case_table_id:
            self.case_table = dm.tables.find(case_table_id)

            foreign_key_case_id = next(
                (
                    item
                    for item in dm.data["foreignKeys"]
                    if item["sourceTableId"] == case_table_id
                    and item["targetTableId"] == activity_table_id
                ),
                None,
            )

            self.activity_case_key = foreign_key_case_id["columns"][0][
                "targetColumnName"
            ]

            self.case_case_key = foreign_key_case_id["columns"][0]["sourceColumnName"]
            self.case_table_name = self.case_table.name
            self._set_dynamic_features_PQL()
            self._set_static_features_PQL()
        else:
            self.case_table = EmptyTable()
            self.case_case_key = ""
            self.case_table_name = ""
            self.activity_case_key = dm.data["processConfigurations"][0]["caseIdColumn"]
            self._set_dynamic_features_PQL()
        self.process_model = ProcessModelFactory.create(
            datamodel=dm, activity_table=self.activity_table_name
        )

    def _set_static_features_PQL(self):
        """Set the static feature names

        :return:
        """
        for attribute in self.case_table.columns:
            if attribute["type"] in self.categorical_types and attribute[
                "name"
            ] not in [self.case_case_key, self.sort_col]:
                self.static_categorical_cols.append(attribute["name"])
            elif attribute["type"] in self.numerical_types and attribute[
                "name"
            ] not in [
                self.case_case_key,
                self.sort_col,
            ]:
                self.static_numerical_cols.append(attribute["name"])

    def _set_dynamic_features_PQL(self):
        """Set the dynamic feature names

        :return:
        """
        for attribute in self.activity_table.columns:
            if attribute["type"] in self.categorical_types and attribute[
                "name"
            ] not in [self.activity_case_key, self.sort_col, self.activity_col]:
                self.dynamic_categorical_cols.append(attribute["name"])
            elif attribute["type"] in self.numerical_types and attribute[
                "name"
            ] not in [self.activity_case_key, self.sort_col]:
                self.dynamic_numerical_cols.append(attribute["name"])

    def _adjust_string_values(self, l: List[str]):
        """Adjust string values for PQL query and dataframe column name.

        :param l: list of original strings
        :return: lists with adjusted strings for PQL and dataframe column name
        """
        list_pql_str = [el.replace("'", "\\'") for el in l]
        list_df_str = [el.replace('"', '\\"') for el in l]

        return list_pql_str, list_df_str

    def get_df_with_filters(self, query: PQL):
        for f in self.filters:
            query.add(f)
        return self.dm.get_data_frame(query, chunksize=self.chunksize)

    def get_valid_vals(
        self,
        table_name: str,
        column_names: List[str],
        min_vals: int = 0,
        max_vals: int = np.inf,
    ):
        """Get values of columns that occur in enough cases. This is done with PQL
        and shall be used before the actual column is queried for the resulting
        DataFrame.

        :param table_name:
        :param column_names:
        :param min_vals:
        :param max_vals:
        :return: dictionary with keys=column names and values: List of the valid
        values in the column.
        """
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
                    query='COUNT_TABLE("' + self.case_table_name + '")',
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

    def get_query_case_ids(self) -> PQLColumn:
        """Get query for case ids

        :return: PQLColumn object with the query
        """
        return PQLColumn(
            name="caseid",
            query='"' + self.case_table_name + '"."' + self.case_case_key + '"',
        )

    def _binarize(self, x: pd.Series, th: int = 1) -> pd.Series:
        """Set all values larger or equal than th to 1, else to 0
        :param x: Series
        :return: binarized Series

        """
        x[x >= th] = 1
        x[x < th] = 0
        return x

    def date_filter_PQL(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ):
        """
        :param start_date: date in the form yyyy-mm-dd
        :param end_date: date in the form yyyy-mm-dd
        :return:
        """
        if start_date:
            date_str_pql = f"{{d'{start_date}'}}"
            filter_str = (
                f'PU_FIRST("{self.case_table_name}", '
                f'"{self.activity_table_name}"."'
                f'{self.eventtime_col}") >= {date_str_pql}'
            )
            filter_start = PQLFilter(filter_str)
            self.filters.append(filter_start)

        if end_date:
            date_str_pql = f"{{d'{end_date}'}}"
            filter_str = (
                f'PU_FIRST("{self.case_table_name}", '
                f'"{self.activity_table_name}"."'
                f'{self.eventtime_col}") <= {date_str_pql}'
            )
            filter_start = PQLFilter(filter_str)
            self.filters.append(filter_start)

    def _gen_dynamic_activity_occurence_attributes(
        self,
        activities: List[str] = None,
        min_vals: int = 0,
        max_vals: int = np.inf,
        is_feature: bool = True,
        is_class_feature: bool = False,
    ) -> List[PreviousActivityOccurrenceAttribute]:
        """Generates the static ActivitiyOccurenceAttributes. If no activities are
        given, all activities are used and it's checked for min and max values. If
        activities are given, it is not checked for min and max occurences."""
        if activities is None:
            activities = self.get_valid_vals(
                table_name=self.process_model.activity_table_str,
                column_names=[self.process_model.activity_column_str],
                min_vals=min_vals,
                max_vals=max_vals,
            )[self.process_model.activity_column_str]

        activity_occ_attributes = []
        for activity in activities:
            attr = PreviousActivityOccurrenceAttribute(
                process_model=self.process_model,
                activity=activity,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            activity_occ_attributes.append(attr)

        return activity_occ_attributes

    def _gen_static_activity_occurence_attributes(
        self,
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
            activities = self.get_valid_vals(
                table_name=self.process_model.activity_table_str,
                column_names=[self.process_model.activity_column_str],
                min_vals=min_vals,
                max_vals=max_vals,
            )[self.process_model.activity_column_str]

        activity_occ_attributes = []
        for activity in activities:
            attr = ActivityOccurenceAttribute(
                process_model=self.process_model,
                activity=activity,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            activity_occ_attributes.append(attr)

        return activity_occ_attributes

    def _gen_static_numeric_activity_table_attributes(
        self,
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
                process_model=self.process_model,
                column_name=column,
                aggregation=aggregation,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            attrs.append(attr)
        return attrs

    def _gen_numeric_case_column_attributes(
        self,
        columns: List[str],
        is_feature: bool = True,
        is_class_feature: bool = False,
    ):
        """Generate list of static CaseTableColumnNumericAttributes

        :param columns:
        :param is_feature:
        :param is_class_feature:
        :return:
        """
        attrs = []
        for column in columns:
            attr = CaseTableColumnNumericAttribute(
                process_model=self.process_model,
                column_name=column,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            attrs.append(attr)
        return attrs

    def _gen_categorical_case_column_attributes(
        self,
        columns: List[str],
        is_feature: bool = True,
        is_class_feature: bool = False,
    ):
        """Generate list of static CaseTableColumnNumericAttributes

        :param columns:
        :param is_feature:
        :param is_class_feature:
        :return:
        """
        attrs = []
        for column in columns:
            attr = CaseTableColumnCategoricalAttribute(
                process_model=self.process_model,
                column_name=column,
                is_feature=is_feature,
                is_class_feature=is_class_feature,
            )
            attrs.append(attr)
        return attrs

    def _default_target_query(self, name="TARGET"):
        """Return PQLColumn that evaluates to 1 for all cases wth default column name
        TARGET"""
        q_str = (
            f'CASE WHEN PU_COUNT("{self.process_model.case_table_str}", '
            f'"{self.process_model.activity_table_str}"."'
            f'{self.process_model.activity_column_str}") >= 1 THEN 1 ELSE 0 END'
        )
        return PQLColumn(name=name, query=q_str)

    def _all_cases_closed_query(self, name="IS_CLOSED"):
        """Return PQLColumn that evaluates to 1 for all cases with default column name
        IS_CLOSED"""
        q_str = (
            f'CASE WHEN PU_COUNT("{self.process_model.case_table_str}", '
            f'"{self.process_model.activity_table_str}"."'
            f'{self.process_model.activity_column_str}") >= 1 THEN 1 ELSE 0 END'
        )
        return PQLColumn(name=name, query=q_str)

    def extract_dfs(
        self,
        static_attributes: List[StaticAttribute],
        dynamic_attributes: List[DynamicAttribute],
        is_closed_indicator: PQLColumn,
        target_variable: PQLColumn,
    ) -> Tuple[Any, ...]:
        # Get PQLColumns from the attributes
        self.process_model.global_filters = self.filters
        static_attributes_pql = [attr.pql_query for attr in static_attributes]
        if dynamic_attributes:
            dynamic_attributes_pql = [attr.pql_query for attr in dynamic_attributes]
            extractor = PQLExtractor(
                process_model=self.process_model,
                target_variable=target_variable,
                is_closed_indicator=is_closed_indicator,
                static_features=static_attributes_pql,
                dynamic_features=dynamic_attributes_pql,
            )
            df_x, df_y = extractor.get_closed_cases(self.dm, only_last_state=False)
        else:
            extractor = StaticPQLExtractor(
                process_model=self.process_model,
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
        # Define closed case (currently define that every case is a closed case)
        is_closed_indicator = self._all_cases_closed_query()

        self.process_model.global_filters = self.filters

        self.transitions_df = extract_transitions(
            process_model=self.process_model,
            dm=self.dm,
            is_closed_indicator=is_closed_indicator,
        )

        static_attributes = [
            WorkInProgressAttribute(
                process_model=self.process_model,
                aggregation="AVG",
                is_feature=True,
                is_class_feature=False,
            ),
            StartActivityAttribute(
                process_model=self.process_model,
                is_feature=True,
                is_class_feature=False,
            ),
            EndActivityAttribute(
                process_model=self.process_model,
                is_feature=True,
                is_class_feature=False,
            ),
            StartActivityTimeAttribute(process_model=self.process_model),
            EndActivityTimeAttribute(process_model=self.process_model)
            # ActivityOccurenceAttribute(process_model=self.process_model,
            #    aggregation="AVG", is_feature=True, is_class_feature=False,
            # )
        ]

        # First see which activities happen often enough to be used. Then create the
        # attributes for those.
        activity_occ_attributes = self._gen_static_activity_occurence_attributes(
            is_feature=True, min_vals=self.min_attr_count, max_vals=self.max_attr_count
        )
        numeric_activity_column_attributes = (
            self._gen_static_numeric_activity_table_attributes(
                columns=self.dynamic_numerical_cols,
                aggregation="AVG",
                is_feature=True,
                is_class_feature=False,
            )
        )
        categorical_case_table_column_attrs = (
            self._gen_categorical_case_column_attributes(
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
            + activity_occ_attributes
            + numeric_activity_column_attributes
            + categorical_case_table_column_attrs
            + numeric_case_table_column_attrs
        )

        self.static_attributes = static_attributes

        dynamic_attributes = []

        self.dynamic_attributes = dynamic_attributes

        self.df_timestamp_column = "Start activity Time"

        target_attribute = CaseDurationAttribute(
            process_model=self.process_model,
            time_aggregation=time_unit,
            is_feature=False,
            is_class_feature=True,
        )

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
            self._gen_categorical_case_column_attributes(
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
            self._gen_categorical_case_column_attributes(
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
