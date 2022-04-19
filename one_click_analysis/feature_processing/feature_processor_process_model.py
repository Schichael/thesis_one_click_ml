from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from prediction_builder.data_extraction import ProcessModelFactory
from pycelonis.celonis_api.event_collection import data_model
from pycelonis.celonis_api.pql.pql import PQL
from pycelonis.celonis_api.pql.pql import PQLColumn
from pycelonis.celonis_api.pql.pql import PQLFilter

from one_click_analysis import utils
from one_click_analysis.errors import NotAValidAttributeError
from one_click_analysis.feature_processing import attributes
from one_click_analysis.feature_processing.attributes import AttributeDataType
from one_click_analysis.feature_processing.attributes_new.static_attributes import (
    CaseDurationAttribute,
    WorkInProgressAttribute,
    ActivityOccurenceAttribute,
)

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
        self.attributes_dict = {}
        self.labels = []
        self.labels_dict = {}
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

    def reset_fp(self):
        """Reset the feature_processor to its initial values."""
        self.__init__(
            self.dm, self.chunksize, self.min_attr_count_perc, self.max_attr_count
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

    def compute_metrics(self, df: pd.DataFrame):
        """Compute metrics between independent variables (the ones stored in
        self.attributes) and dependent variables (the ones stored in self.labels). The
        computed metrics are stored within the Attribute objects in self.attributes

        :param df: DataFrame with attributes
        :return:
        """

        for attr in self.attributes:
            # For numerical attribute only compute the correlation coefficient
            if attr.attribute_data_type != attributes.AttributeDataType.NUMERICAL:
                # Number of cases with this attribute
                attr.cases_with_attribute = len(
                    df[df[attr.df_attribute_name] == 1].index
                )

                # Influence on dependent variable
                label_influences = []
                for label in self.labels:
                    label_val_0 = df[df[attr.df_attribute_name] == 0][
                        label.df_attribute_name
                    ].mean()
                    label_val_1 = df[df[attr.df_attribute_name] == 1][
                        label.df_attribute_name
                    ].mean()
                    label_influences.append(label_val_1 - label_val_0)

                attr.label_influence = label_influences

            # Correlation coefficient
            attribute_df = pd.DataFrame(df[attr.df_attribute_name])
            correlations = []
            for label in self.labels:
                label_series = df[label.df_attribute_name]
                correlation = attribute_df.corrwith(label_series)[
                    attr.df_attribute_name
                ]
                correlations.append(correlation)
            attr.correlation = correlations

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
        cols = df_val_counts.columns

        valid_vals_dict = {}
        for col in cols:
            if ~col.endswith("_values"):
                continue
            col_name = col[:-7]
            count_col_name = col_name + "_count"
            valid_vals_dict[col_name] = df_val_counts[
                (df_val_counts[count_col_name] >= min_vals)
                & (df_val_counts[count_col_name] <= max_vals)
            ][col].values.tolist()

        return valid_vals_dict

    def one_hot_encoding_PQL(
        self,
        table: str,
        column_names: List[str],
        major_attribute: attributes.MajorAttribute,
        minor_attribute: attributes.MinorAttribute,
        attribute_datatype: attributes.AttributeDataType,
        min_vals: int = 1,
        max_vals: int = np.inf,
        suffix: str = "",
        prefix: str = "",
        binarize_threshold: Optional[int] = None,
        is_attr: bool = False,
        is_label: bool = False,
    ) -> PQL:
        """Perform one-hot like encoding. The column name will be <table name>.<column
        value>. A prefix and a suffix can be added optionally. The resulting values
        will be the number of occurrences per case if the table is the activity
        table. If the table is the case table, the entries will be 1(attribute
        occurring) or 0 (attribute not occurring)

        :param table: Celonis table name (name of case table or activity table)
        :param column_names: list of column names to one hot encode
        :param major_attribute: MajorAttribute of the resulting attributes
        :param minor_attribute: MinorAttribute of the resulting attributes
        :param attribute_datatype: The datatype of the attribute
        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :param suffix: suffix to the column name
        :param prefix: prefix to the column name
        :param binarize_threshold: if binarize_threshold is an integer, the result
        will be binarized. Results with a value >= binarize_threshold will become 1
        everything else will become 0.
        :return: PQL object with the query
        """
        if suffix != "":
            suffix = " " + suffix
        if prefix != "":
            prefix = " " + prefix
        query = PQL()
        # query.add(self.get_query_case_ids())
        for column_name in column_names:
            # Get counts of unique values
            query_unique = PQL()
            query_unique.add(
                PQLColumn(
                    name="values",
                    query='DISTINCT("' + table + '"."' + column_name + '")',
                )
            )
            query_unique.add(
                PQLColumn(
                    name="count", query='COUNT_TABLE("' + self.case_table_name + '")'
                )
            )
            # Can just query enough occurences using filter

            df_unique_vals = self.get_df_with_filters(query_unique)

            # Remove too few and too many counts
            df_unique_vals = df_unique_vals[
                (df_unique_vals["count"] >= min_vals)
                & (df_unique_vals["count"] <= max_vals)
            ]
            unique_values = list(df_unique_vals["values"])
            # Remove None values
            unique_values = [x for x in unique_values if x is not None]

            # Add escaping characters
            list_pql_str, list_df_str = self._adjust_string_values(unique_values)

            for pql_str, df_str in zip(list_pql_str, list_df_str):
                df_attr_name = (
                    prefix + table + "_" + column_name + " = " + df_str + suffix
                )
                display_name = (
                    prefix + table + "." + column_name + " = " + df_str + suffix
                )

                major_attribute_type = major_attribute
                minor_attribute_type = minor_attribute

                query_val = (
                    'SUM(CASE WHEN "'
                    + table
                    + '"."'
                    + column_name
                    + "\" = '"
                    + pql_str
                    + "' THEN 1 ELSE 0 "
                    "END)"
                )

                if binarize_threshold is not None:
                    query_val = (
                        "CASE WHEN("
                        + query_val
                        + " >= "
                        + str(binarize_threshold)
                        + ") THEN 1 ELSE 0 END"
                    )

                attr_obj = attributes.Attribute(
                    major_attribute_type,
                    minor_attribute_type,
                    attribute_datatype,
                    df_attr_name,
                    display_name,
                    query_val,
                    column_name=column_name,
                )
                if is_attr:
                    self.attributes.append(attr_obj)
                    self.attributes_dict[df_attr_name] = attr_obj
                if is_label:
                    self.labels.append(attr_obj)
                    self.labels_dict[df_attr_name] = attr_obj

                query.add(PQLColumn(name=df_attr_name, query=query_val))
        return query
        # dataframe = self.dm.get_data_frame(query, chunksize=self.chunksize)
        # return dataframe

    def aggregate_static_categorical_PQL(
        self,
        min_vals: int = 0,
        max_vals: int = np.inf,
        is_attr: bool = False,
        is_label: bool = False,
    ) -> PQL:
        """Aggregate static categorical attributes (categorical columns in the case
        table). This is done using one-hot-encoding

        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :return: PQL object with the query
        """
        major_attribute = attributes.MajorAttribute.CASE
        minor_attribute = attributes.CaseTableColumnMinorAttribute()
        # Need to add binarize_threshold=1 her because else we get the number of
        # events in a case instead of 1 if the column value occurs in the case
        query_categorical = self.one_hot_encoding_PQL(
            self.case_table_name,
            self.static_categorical_cols,
            major_attribute,
            minor_attribute,
            attributes.AttributeDataType.CATEGORICAL,
            min_vals=min_vals,
            max_vals=max_vals,
            binarize_threshold=1,
            is_attr=is_attr,
            is_label=is_label,
        )

        # df_static_categorical = self._conv_dtypes_PQL(
        #    df_static_categorical, ["object"], "category"
        # )
        return query_categorical

    def get_static_numerical_PQL(self) -> PQL:
        """Get static numerical attributes (numerical columns in the case table). The
        numerical columns are simply fetched from the database.

        :return: PQL object with the query
        """
        query = PQL()
        # query.add(self.get_query_case_ids())
        for attribute in self.static_numerical_cols:
            df_attr_name = self.case_table_name + "." + attribute
            display_name = self.case_table_name + "." + attribute

            query_attr = '"' + self.case_table_name + '".' + '"' + attribute + '"'

            query.add(PQLColumn(name=df_attr_name, query=query_attr))
            attr_obj = attributes.Attribute(
                attributes.MajorAttribute.ACTIVITY,
                attributes.CaseTableColumnMinorAttribute(),
                attributes.AttributeDataType.NUMERICAL,
                df_attr_name,
                display_name,
                query_attr,
                column_name=attribute,
            )
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj
        # dataframe = self.dm.get_data_frame(query, chunksize=self.chunksize)
        return query

    def aggregate_dynamic_numerical_PQL(
        self,
        aggregations: Optional[List[str]] = None,
        is_attr: bool = False,
        is_label: bool = False,
    ) -> PQL:
        """Aggregate dynamic numerical columns (numerical columns in the activity
        table).

        :param aggregations: List of aggregations to use. The names of the
        aggregations need to be as used in PQL. If aggregations is None, 'AVG' is
        used which computes the average value within a case.
        :return: PQL object with the query
        """
        if aggregations is None:
            aggregations = ["AVG"]
        query = PQL()
        # query.add(self.get_query_case_ids())
        for agg in aggregations:
            for attribute in self.dynamic_numerical_cols:
                df_attr_name = self.activity_table_name + "." + agg + "_" + attribute
                display_name = (
                    self.activity_table_name
                    + "."
                    + attribute
                    + " ("
                    + self.get_aggregation_display_name(agg)
                    + ")"
                )
                query_att = (
                    agg
                    + '("'
                    + self.activity_table_name
                    + '".'
                    + '"'
                    + attribute
                    + '")'
                )
                query.add(PQLColumn(name=df_attr_name, query=query_att))
                attr_obj = attributes.Attribute(
                    attributes.MajorAttribute.ACTIVITY,
                    attributes.ActivityTableColumnMinorAttribute(agg),
                    attributes.AttributeDataType.NUMERICAL,
                    df_attr_name,
                    display_name,
                    query_att,
                    column_name=attribute,
                )
                if is_attr:
                    self.attributes.append(attr_obj)
                    self.attributes_dict[df_attr_name] = attr_obj
                if is_label:
                    self.labels.append(attr_obj)
                    self.labels_dict[df_attr_name] = attr_obj
        # dataframe = self.dm.get_data_frame(query, chunksize=self.chunksize)

        return query

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

    def _gen_activity_occurence_attributes(self, is_feature: bool = True,
                                           is_class_feature: bool = False) -> List[ActivityOccurenceAttribute]:
        """Generates the static ActivitiyOccurenceAttributes for the valid
        activities. Valid = Not too many and not too few occurences."""
        valid_activities = self.get_valid_vals(
            table_name=self.process_model.activity_table_str,
            column_names=[self.process_model.activity_column_str],
            min_vals=self.min_attr_count, max_vals=self.max_attr_count)[
            self.process_model.activity_column_str]

        activity_occ_attributes = []
        for activity in valid_activities:
            attr = ActivityOccurenceAttribute(process_model=self.process_model,
                activity=activity, is_feature=is_feature,
                                       is_class_feature=is_class_feature,
             )
            activity_occ_attributes.append(attr)

        return activity_occ_attributes


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

        static_attributes = [
            CaseDurationAttribute(
                process_model=self.process_model,
                time_aggregation=time_unit,
                is_feature=False,
                is_class_feature=True,
            ),
            WorkInProgressAttribute(
                process_model=self.process_model,
                aggregation="AVG",
                is_feature=True,
                is_class_feature=False,
            ),
            # ActivityOccurenceAttribute(process_model=self.process_model,
            #    aggregation="AVG", is_feature=True, is_class_feature=False,
            # )
        ]

        # First see which activities happen often enough to be used. Then create the
        # attributes for those.

        activity_occ_attributes = self._gen_activity_occurence_attributes(
            is_feature=True)

        static_attributes = static_attributes + activity_occ_attributes

        minor_attrs = [
            attributes.StartActivityMinorAttribute(is_attr=True),
            attributes.EndActivityMinorAttribute(is_attr=True),
            attributes.ActivityOccurenceMinorAttribute(is_attr=True),
            attributes.ReworkOccurenceMinorAttribute(is_attr=True),
            attributes.WorkInProgressMinorAttribute(aggregations=["AVG"], is_attr=True),
            attributes.CaseTableColumnMinorAttribute(is_attr=True),
            attributes.ActivityTableColumnMinorAttribute(
                aggregations=["AVG"], is_attr=True
            ),
            attributes.CaseDurationMinorAttribute(
                time_aggregation=time_unit, is_label=True
            ),
        ]

        df = self.process_attributes(minor_attrs)

        # Additional columns
        query_additional = PQL()
        query_additional.add(self.get_query_case_ids())
        query_additional.add(self.start_activity_time_PQL())
        query_additional.add(self.end_activity_time_PQL())
        df_additional = self.get_df_with_filters(query_additional)

        df_joined = utils.join_dfs([df, df_additional], keys=["caseid"] * 2)

        self.df = df_joined
        self.post_process()

    def transition_occurence_PQL(
        self,
        transitions: List[Tuple[str, List[str]]],
        is_attr: bool = False,
        is_label: bool = False,
    ):

        query = PQL()
        for transition in transitions:
            start_activity = transition[0]
            end_activities = transition[1]

            for end_activity in end_activities:
                df_attr_name = "Next Activity = " + end_activity
                attr_obj = attributes.Attribute(
                    major_attribute_type=attributes.MajorAttribute.ACTIVITY,
                    minor_attribute_type=attributes.TransitionMinorAttribute(
                        transitions, is_label=is_label
                    ),
                    attribute_data_type=attributes.AttributeDataType.CATEGORICAL,
                    df_attribute_name=df_attr_name,
                    display_name=df_attr_name,
                    query="",
                )

                if is_label:
                    self.labels.append(attr_obj)
                    self.labels_dict[df_attr_name] = attr_obj
                if is_attr:
                    self.attributes.append(attr_obj)
                    self.attributes_dict[df_attr_name] = attr_obj

                q_str = (
                    f"CASE WHEN PROCESS EQUALS '{start_activity}' TO "
                    f"'{end_activity}' THEN 1 ELSE 0 END"
                )

                query.add(PQLColumn(q_str, df_attr_name))

        return query

    def filter_transition_PQL(self, start_activity: str):
        # term_end_activities = (
        #    "(" + ", ".join("'" + act + "'" for act in end_activities) + ")"
        # )
        filter_str = f"PROCESS EQUALS '{start_activity}'"
        return PQLFilter(filter_str)

    def run_decision_point_PQL(
        self,
        start_activity: str,
        end_activities: List[str],
        time_unit="DAYS",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Fetching data and preprocessing for the decision point analysis. Currently
        only static attributes are supported

        :param start_activity:
        :param end_activities:
        :param time_unit:
        :return:
        """
        # Add filters for start and end date
        self.date_filter_PQL(start_date, end_date)

        # Make label a list
        self.labels = []

        # add filter
        self.filters.append(self.filter_transition_PQL(start_activity))

        # Get the attributes
        minor_attrs = [
            attributes.StartActivityMinorAttribute(is_attr=True),
            # attributes.EndActivityMinorAttribute(is_attr=True),
            attributes.CaseTableColumnMinorAttribute(is_attr=True),
            attributes.ActivityTableColumnMinorAttribute(
                aggregations="AVG", is_attr=True
            ),
            attributes.TransitionMinorAttribute(
                transitions=[(start_activity, end_activities)], is_label=True
            ),
            attributes.CaseDurationMinorAttribute(
                time_aggregation=time_unit, is_attr=False
            ),
        ]

        df = self.process_attributes(minor_attrs)

        # Additional columns
        query_additional = PQL()
        query_additional.add(self.get_query_case_ids())
        query_additional.add(self.start_activity_time_PQL())
        query_additional.add(self.end_activity_time_PQL())
        df_additional = self.get_df_with_filters(query_additional)

        df_joined = utils.join_dfs([df, df_additional], keys=["caseid"] * 2)
        self.df = df_joined
        self.post_process()

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

    def post_process(self):
        """postprocess DataFrame. numberical nans are replaced by median value,
        categorical nans are replaced by most occuring value.

        :return:
        """
        categorical_df_cols = [
            x.df_attribute_name
            for x in self.attributes
            if x.attribute_data_type == AttributeDataType.CATEGORICAL
        ]

        numerical_df_cols = [
            x.df_attribute_name
            for x in self.attributes
            if x.attribute_data_type == AttributeDataType.NUMERICAL
        ]

        self.df.loc[:, categorical_df_cols] = self.df[categorical_df_cols].fillna(
            self.df[categorical_df_cols].mode()
        )

        self.df.loc[:, numerical_df_cols] = self.df[numerical_df_cols].fillna(
            self.df[numerical_df_cols].median()
        )

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
