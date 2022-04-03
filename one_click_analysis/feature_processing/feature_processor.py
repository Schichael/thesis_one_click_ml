from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from pycelonis.celonis_api.event_collection import data_model
from pycelonis.celonis_api.pql.pql import PQL
from pycelonis.celonis_api.pql.pql import PQLColumn

from one_click_analysis import utils
from one_click_analysis.errors import NotAValidAttributeError
from one_click_analysis.feature_processing import attributes

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
        self.label = None
        self.label_dict = {}
        self._init_datamodel(self.dm)
        (
            self.min_attr_count,
            self.max_attr_count,
        ) = self.compute_min_max_attribute_counts_PQL(
            min_attr_count_perc, max_attr_count_perc
        )
        self.df = None
        self.minor_attrs = []

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

    def get_aggregation_display_name(self, agg):
        """Generate the name of the aggregation to display to the user from the
        aggregation String that is used for a PQL query

        :param agg: original aggregation name as used for
        :return: aggregation string to display
        """
        if agg == "MIN":
            return "minimum"
        elif agg == "MAX":
            return "maximum"
        elif agg == "AVG":
            return "mean"
        elif agg == "MEDIAN":
            return "median"
        elif agg == "FIRST":
            return "first"
        elif agg == "LAST":
            return "last"

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
        self.attributes) and dependent variable (the one stored in self.label). The
        computed metrics are stored within the Attribute objects in self.attributes

        :param df: DataFrame with attributes
        :return:
        """

        for attr in self.attributes:
            # For numerical attribute only compute the correlation coefficient
            if attr.attribute_data_type != attributes.AttributeDataType.NUMERICAL:
                # Influence on dependent variable
                label_val_0 = df[df[attr.df_attribute_name] == 0][
                    self.label.df_attribute_name
                ].mean()
                label_val_1 = df[df[attr.df_attribute_name] == 1][
                    self.label.df_attribute_name
                ].mean()
                attr.label_influence = label_val_1 - label_val_0

                # Number of cases with this attribute
                attr.cases_with_attribute = len(
                    df[df[attr.df_attribute_name] == 1].index
                )

            # Correlation coefficient
            label_series = df[self.label.df_attribute_name]
            attribute_df = pd.DataFrame(df[attr.df_attribute_name])
            correlations = attribute_df.corrwith(label_series)
            attr.correlation = correlations[attr.df_attribute_name]

    def one_hot_encoding_PQL(
        self,
        table: str,
        column_names: List[str],
        major_attribute: attributes.MajorAttribute,
        minor_attribute: attributes.MinorAttribute,
        min_vals: int = 1,
        max_vals: int = np.inf,
        suffix: str = "",
        prefix: str = "",
        binarize_threshold: Optional[int] = None,
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
        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :param suffix: suffix to the column name
        :param prefix: prefix to the column name
        :param binarize_threshold: if binarize_threshold is an integer, the result
        will be binarized. Results with a value >= binarize_threshold will become 1
        everything else will become 0.
        :return: Dataframe with the one hot encoded attributes
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

            df_unique_vals = self.dm.get_data_frame(
                query_unique, chunksize=self.chunksize
            )
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
                    attributes.AttributeDataType.CATEGORICAL,
                    df_attr_name,
                    display_name,
                    query_val,
                    column_name=column_name,
                )
                self.attributes.append(attr_obj)
                self.attributes_dict[df_attr_name] = attr_obj

                query.add(PQLColumn(name=df_attr_name, query=query_val))
        return query
        # dataframe = self.dm.get_data_frame(query, chunksize=self.chunksize)
        # return dataframe

    def aggregate_static_categorical_PQL(
        self, min_vals: int = 0, max_vals: int = np.inf
    ) -> PQL:
        """Aggregate static categorical attributes (categorical columns in the case
        table). This is done using one-hot-encoding

        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :return: Dataframe with the one hot encoded attributes
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
            min_vals=min_vals,
            max_vals=max_vals,
            binarize_threshold=1,
        )

        # df_static_categorical = self._conv_dtypes_PQL(
        #    df_static_categorical, ["object"], "category"
        # )
        return query_categorical

    def get_static_numerical_PQL(self) -> PQL:
        """Get static numerical attributes (numerical columns in the case table). The
        numerical columns are simply fetched from the database.

        :return: DataFrame with static numerical attributes
        """
        query = PQL()
        # query.add(self.get_query_case_ids())
        for attribute in self.static_numerical_cols:
            df_attr_name = self.case_table_name + "_" + attribute
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

    def aggregate_dynamic_categorical_PQL(
        self, min_vals: int = 1, max_vals: int = np.inf
    ) -> PQL:
        """Aggregate dynamic categorical columns (categorical columns in the activity
        table). This is done using one-hot-encoding.

        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :return: DataFrame with the dynamic categorical attributes
        """
        major_attribute = attributes.MajorAttribute.ACTIVITY
        minor_attribute = attributes.ActivityTableColumnMinorAttribute()
        query_categorical = self.one_hot_encoding_PQL(
            self.activity_table_name,
            self.dynamic_categorical_cols,
            major_attribute,
            minor_attribute,
            min_vals=min_vals,
            max_vals=max_vals,
            binarize_threshold=1,
        )

        # df_dynamic_categorical = self._conv_dtypes_PQL(
        #    df_dynamic_categorical, ["object"], "category"
        # )
        return query_categorical

    def aggregate_dynamic_numerical_PQL(
        self, aggregations: Optional[List[str]] = None
    ) -> PQL:
        """Aggregate dynamic numerical columns (numerical columns in the activity
        table).

        :param aggregations: List of aggregations to use. The names of the
        aggregations need to be as used in PQL. If aggregations is None, 'AVG' is
        used which computes the average value within a case.
        :return: DataFrame with the aggregated dynamic numerical attributes
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
                self.attributes.append(attr_obj)
                self.attributes_dict[df_attr_name] = attr_obj
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

    def one_hot_encode_special(
        self,
        query_str,
        attribute_name,
        major_attribute: attributes.MajorAttribute,
        minor_attribute: attributes.MinorAttribute,
        attribute_data_type: attributes.AttributeDataType,
        min_vals: int = 0,
        max_vals: int = np.inf,
        binarize_threshold: Optional[int] = None,
    ) -> PQL:
        """One hot encoding with a special query. The resulting values will be the
        number of occurences per case if the table is the activity table.
        query_str is string with what comes within the DISTINCT() brackets in the
        frist query and then in the CASE WHEN in the second query.

        :param query_str: the query
        :param attribute_name: the attribute name in "attribute_name = value"
        :param major_attribute: the major attribute
        :param minor_attribute: the minor attribute
        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :param binarize_threshold: if binarize_threshold is an integer, the result
        will be binarized. Results with a value >= binarize_threshold will become 1
        everything else will become 0.
        :return: DataFrame with the attributes gotten from the query
        """

        query_unique = PQL()
        query_unique.add(PQLColumn(name="values", query="DISTINCT(" + query_str + ")"))
        query_unique.add(
            PQLColumn(
                name="count",
                query='COUNT(DISTINCT "'
                + self.activity_table_name
                + '"."'
                + self.activity_case_key
                + '")',
            )
        )

        df_unique_vals = self.dm.get_data_frame(query_unique, chunksize=self.chunksize)
        # remove too few counts
        df_unique_vals = df_unique_vals[
            (df_unique_vals["count"] >= min_vals)
            & (df_unique_vals["count"] <= max_vals)
        ]
        unique_values = list(df_unique_vals["values"])
        # Remove None values
        unique_values = [x for x in unique_values if x is not None]
        # Add escaping characters
        list_pql_str, list_df_str = self._adjust_string_values(unique_values)
        query = PQL()
        # query.add(self.get_query_case_ids())
        for pql_str, df_str in zip(list_pql_str, list_df_str):
            df_attr_name = attribute_name + " = " + df_str
            display_name = attribute_name + " = " + df_str
            query_attr = (
                "SUM(CASE WHEN " + query_str + " = " + "'" + pql_str + "' THEN 1 "
                "ELSE 0 "
                "END)"
            )
            if binarize_threshold is not None:
                query_attr = (
                    "CASE WHEN("
                    + query_attr
                    + " >= "
                    + str(binarize_threshold)
                    + ") THEN 1 ELSE 0 END"
                )
            query.add(PQLColumn(name=df_attr_name, query=query_attr))
            attr_obj = attributes.Attribute(
                major_attribute,
                minor_attribute,
                attribute_data_type,
                df_attr_name,
                display_name,
                query_attr,
            )
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj
        # dataframe = self.dm.get_data_frame(query, chunksize=self.chunksize)
        return query

    def start_activity_PQL(self, min_vals: int = 0, max_vals: int = np.inf) -> PQL:
        """Get the case start activities as one-hot-encoded attributes

        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :return: DataFrame with the start activities
        """
        attribute_name = "Start activity"
        attribute_data_type = attributes.AttributeDataType.CATEGORICAL
        major_attribute = attributes.MajorAttribute.ACTIVITY
        minor_attribute = attributes.StartActivityMinorAttribute()
        query_str = (
            'PU_FIRST("' + self.case_table_name + '", '
            '"' + self.activity_table_name + '"."' + self.activity_col + '")'
        )
        # Need to add binarize_threshold=1 her because else we get the number of
        # events in a case instead of 1 if the first activity is the selected activity
        query = self.one_hot_encode_special(
            query_str,
            attribute_name,
            major_attribute,
            minor_attribute,
            attribute_data_type,
            min_vals,
            max_vals,
            binarize_threshold=1,
        )

        return query

    def end_activity_PQL(self, min_vals: int = 0, max_vals: int = np.inf) -> PQL:
        """Get the case end activities as one-hot-encoded attributes

        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :return: DataFrame with the end activities
        """
        attribute_name = "End activity"
        attribute_data_type = attributes.AttributeDataType.CATEGORICAL
        major_attribute = attributes.MajorAttribute.ACTIVITY
        minor_attribute = attributes.EndActivityMinorAttribute()
        query_str = (
            'PU_LAST("' + self.case_table_name + '", '
            '"' + self.activity_table_name + '"."' + self.activity_col + '")'
        )
        # Need to add binarize_threshold=1 her because else we get the number of
        # events in a case instead of 1 if the last activity is the selected activity
        query = self.one_hot_encode_special(
            query_str,
            attribute_name,
            major_attribute,
            minor_attribute,
            attribute_data_type,
            min_vals,
            max_vals,
            binarize_threshold=1,
        )
        return query

    def start_activity_time_PQL(self) -> PQL:
        """Get datetime of the first activity in a case

        :return: DataFrame with the start activity times
        """

        query_str = (
            'PU_FIRST("' + self.case_table_name + '", '
            '"' + self.activity_table_name + '"."' + self.eventtime_col + '")'
        )
        query = PQL()
        # query.add(self.get_query_case_ids())
        query.add(PQLColumn(name="Case start time", query=query_str))
        # df = self.dm.get_data_frame(query, chunksize=self.chunksize)
        return query

    def end_activity_time_PQL(self) -> PQL:
        """Get datetime of the last activity in a case

        :return: DataFrame with the end activity times
        """
        query_str = (
            'PU_LAST("' + self.case_table_name + '", '
            '"' + self.activity_table_name + '"."' + self.eventtime_col + '")'
        )
        query = PQL()
        # query.add(self.get_query_case_ids())
        query.add(PQLColumn(name="Case end time", query=query_str))
        # df = self.dm.get_data_frame(query, chunksize=self.chunksize)
        return query

    def _binarize(self, x: pd.Series, th: int = 1) -> pd.Series:
        """Set all values larger or equal than th to 1, else to 0
        :param x: Series
        :return: binarized Series

        """
        x[x >= th] = 1
        x[x < th] = 0
        return x

    def binary_activity_occurence_PQL(
        self, min_vals: int = 0, max_vals: int = np.inf
    ) -> PQL:
        """Get binary activity occurences as one-hot-encoded attributes.

        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :return: DataFrame wuth the binary activity occurences
        """
        suffix = "(occurence)"
        major_attribute = attributes.MajorAttribute.ACTIVITY
        minor_attribute = attributes.ActivityOccurenceMinorAttribute()
        query = self.one_hot_encoding_PQL(
            table=self.activity_table_name,
            column_names=[self.activity_col],
            major_attribute=major_attribute,
            minor_attribute=minor_attribute,
            min_vals=min_vals,
            max_vals=max_vals,
            suffix=suffix,
            binarize_threshold=1,
        )
        # Remove values with too few occurences per case key, can this be done in PQL
        # directly???

        # df_activities[df_activities.drop("caseid", axis=1).columns] = df_activities[
        #    df_activities.drop("caseid", axis=1).columns
        # ].apply(lambda x: self._binarize(x, 1), axis=1)
        # df_activities = self._conv_dtypes_PQL(df_activities, ["object"], "category")
        return query

    def binary_rework_PQL(self, min_vals: int = 0, max_vals: int = np.inf) -> PQL:
        """Get binary rework as one-hot-encoded attributes(1 if rework of an activity
        happens, 0 if not).

        :param min_vals: minimum number of cases with attribute to consider
        :param max_vals: maximum number of cases with attribute to consider
        :return: DataFrame with the binary rework occurences
        """
        suffix = "(rework)"
        major_attribute = attributes.MajorAttribute.ACTIVITY
        minor_attribute = attributes.ReworkOccurenceMinorAttribute()

        query = self.one_hot_encoding_PQL(
            table=self.activity_table_name,
            column_names=[self.activity_col],
            major_attribute=major_attribute,
            minor_attribute=minor_attribute,
            min_vals=min_vals,
            max_vals=max_vals,
            suffix=suffix,
            binarize_threshold=2,
        )
        # Remove values with too few occurences per case key, can this be done in PQL
        # directly???
        # df_activities[df_activities.drop("caseid", axis=1).columns] = df_activities[
        #    df_activities.drop("caseid", axis=1).columns
        # ].apply(lambda x: self._binarize(x, 2), axis=1)
        # remove attributes with too few 1 values

        # df_activities = self._remove_rare_or_too_many(df_activities, min_vals)

        # df_activities = self._conv_dtypes_PQL(df_activities, ["object"], "category")
        return query

    def num_events_PQL(self) -> PQL:
        """Get number of events in a case

        :return: DataFrame with the number of events in a case
        """
        df_attr_name = "Event count"
        display_name = "Event count"
        major_attribute = attributes.MajorAttribute.ACTIVITY

        q_num_events = (
            'PU_COUNT("' + self.case_table_name + '", '
            '"' + self.activity_table_name + '"."' + self.activity_col + '")'
        )
        query = PQL()
        # query.add(self.get_query_case_ids())
        query.add(PQLColumn(name="num_events", query=q_num_events))
        attr_obj = attributes.Attribute(
            major_attribute,
            attributes.EventCountMinorAttribute(),
            attributes.AttributeDataType.NUMERICAL,
            df_attr_name,
            display_name,
            q_num_events,
        )
        self.attributes.append(attr_obj)
        self.attributes_dict[df_attr_name] = attr_obj
        # df = self.dm.get_data_frame(query, chunksize=self.chunksize)
        return query

    def work_in_progress_PQL(self, aggregations: Optional[List[str]] = None) -> PQL:
        """Get the work in progress PQL query

        :param aggregations: List of PQL aggregations to use. If None, 'AVG' is used.
        :return: DataFrame with the work in progress in a case
        """

        if aggregations is None:
            aggregations = ["AVG"]

        query = PQL()
        # query.add(self.get_query_case_ids())

        for agg in aggregations:
            agg_display_name = self.get_aggregation_display_name(agg)
            df_attr_name = "Work in progress" + " (" + agg_display_name + ")"
            display_name = "Work in progress" + " (" + agg_display_name + ")"
            major_attribute = attributes.MajorAttribute.CASE

            q = (
                "PU_" + agg + ' ( "' + self.case_table_name + '", RUNNING_SUM( '
                "CASE WHEN "
                'INDEX_ACTIVITY_ORDER ( "'
                + self.activity_table_name
                + '"."'
                + self.activity_col
                + '" ) = 1 THEN 1 WHEN '
                'INDEX_ACTIVITY_ORDER_REVERSE ( "'
                + self.activity_table_name
                + '"."'
                + self.activity_col
                + '" ) = 1 THEN -1 ELSE 0 END, ORDER BY ( "'
                + self.activity_table_name
                + '"."'
                + self.eventtime_col
                + '" ) ) )'
            )
            query.add(PQLColumn(name=df_attr_name, query=q))
            attr_obj = attributes.Attribute(
                major_attribute,
                attributes.WorkInProgressMinorAttribute(agg),
                attributes.AttributeDataType.NUMERICAL,
                df_attr_name,
                display_name,
                q,
            )
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj
        # df = self.dm.get_data_frame(query, chunksize=self.chunksize)

        return query

    def case_duration_PQL(self, time_aggregation, is_label: bool = False) -> PQL:
        """Get total case time.

        :param time_aggregation:
        :param is_label: whether the attribute is used as the label(dependent
        variable) or not
        :return: DataFrame with total case times
        """
        df_attr_name = "case duration"
        display_name = "case duration"
        major_attribute = attributes.MajorAttribute.CASE
        minor_attribute = attributes.CaseDurationMinorAttribute(time_aggregation)

        query = PQL()
        # query.add(
        #    PQLColumn(
        #        name="caseid",
        #        query='"' + self.case_table_name + '"."' + self.case_case_key + '"',
        #    )
        # )
        q_total_time = (
            "(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO ALL_OCCURRENCE["
            "'Process End'], REMAP_TIMESTAMPS(\""
            + self.activity_table_name
            + '"."'
            + self.eventtime_col
            + '", '
            + time_aggregation
            + ")))"
        )
        attr_obj = attributes.Attribute(
            major_attribute,
            minor_attribute,
            attributes.AttributeDataType.NUMERICAL,
            df_attr_name,
            display_name=display_name,
            query=q_total_time,
            unit=time_aggregation.lower(),
        )
        if is_label:
            self.label = attr_obj
            self.label_dict[df_attr_name] = attr_obj
        else:
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj
        query.add(PQLColumn(q_total_time, "case duration"))
        # dataframe = self.dm.get_data_frame(query, chunksize=self.chunksize)
        return query

    def process_attributes(self, attrs: List[attributes.MinorAttribute]):
        """Generate attributes from the list attrs.

        :param attrs: List with MinorAttribute objects.
        :return:
        """
        # dfs = []
        query = PQL()
        query.add(self.get_query_case_ids())
        for attr in attrs:
            print(f"Attribute fetching: {attr.attribute_name}")
            if not attr.is_label:
                self.minor_attrs.append(attr)
            query.add(self.get_attr_df(attr))

        df = self.dm.get_data_frame(query, chunksize=self.chunksize)
        # joined_df = utils.join_dfs(dfs, keys=["caseid"] * len(dfs))
        self.compute_metrics(df)
        return df

    def run_total_time_PQL(self, time_unit="DAYS"):
        """Run feature processing for total case time analysis.

        :param time_unit: time unit to use. E.g. DAYS if attributes shall be in days.
        :return: DataFrame with the processed attributes.
        """
        minor_attrs = [
            attributes.StartActivityMinorAttribute(),
            attributes.EndActivityMinorAttribute(),
            attributes.ActivityOccurenceMinorAttribute(),
            attributes.ReworkOccurenceMinorAttribute(),
            attributes.WorkInProgressMinorAttribute(aggregations=["AVG"]),
            attributes.CaseTableColumnMinorAttribute(),
            attributes.ActivityTableColumnMinorAttribute(aggregations=["AVG"]),
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
        df_additional = self.dm.get_data_frame(
            query_additional, chunksize=self.chunksize
        )

        df_joined = utils.join_dfs([df, df_additional], keys=["caseid"] * 2)

        self.df = df_joined

        """
        start_activity_time_df = self.start_activity_time_PQL()
        end_activity_time_df = self.end_activity_time_PQL()
        #start_activity_df = self.start_activity_PQL(
        #    self.min_attr_count, self.max_attr_count
        #)
        #end_activity_df = self.end_activity_PQL(
        #    self.min_attr_count, self.max_attr_count
        #)
        #binary_activity_occurence_df = self.binary_activity_occurence_PQL(
        #    self.min_attr_count, self.max_attr_count
        #)
        #binary_rework_df = self.binary_rework_PQL(
        #    self.min_attr_count, self.max_attr_count
        #)
        #work_in_progress_df = self.work_in_progress_PQL(aggregations=["AVG"])

        static_cat_df = self.aggregate_static_categorical_PQL(
            self.min_attr_count, self.max_attr_count
        )
        static_num_df = self.get_static_numerical_PQL()
        dyn_cat_df = self.aggregate_dynamic_categorical_PQL(
            self.min_attr_count, self.max_attr_count
        )
        dyn_num_df = self.aggregate_dynamic_numerical_PQL()
        total_time_df = self.case_duration_PQL(time_unit, is_label=True)
        joined_df = utils.join_dfs(
            [
                start_activity_time_df,
                end_activity_time_df,
                start_activity_df,
                end_activity_df,
                binary_activity_occurence_df,
                binary_rework_df,
                work_in_progress_df,
                static_cat_df,
                static_num_df,
                dyn_cat_df,
                dyn_num_df,
                total_time_df,
            ],
            keys=["caseid"] * 12,
        )
        self.compute_metrics(joined_df)
        self.df = joined_df
        """

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

    def get_attr_df(self, attr: attributes.MinorAttribute) -> pd.DataFrame:
        if isinstance(attr, attributes.CaseDurationMinorAttribute):
            df = self.case_duration_PQL(attr.time_aggregation, is_label=attr.is_label)
        elif isinstance(attr, attributes.WorkInProgressMinorAttribute):
            df = self.work_in_progress_PQL(utils.make_list(attr.aggregations))
        elif isinstance(attr, attributes.EventCountMinorAttribute):
            df = self.num_events_PQL()
        elif isinstance(attr, attributes.ReworkOccurenceMinorAttribute):
            df = self.binary_rework_PQL(self.min_attr_count, self.max_attr_count)
        elif isinstance(attr, attributes.ActivityOccurenceMinorAttribute):
            df = self.binary_activity_occurence_PQL(
                self.min_attr_count, self.max_attr_count
            )
        elif isinstance(attr, attributes.EndActivityMinorAttribute):
            df = self.end_activity_PQL(self.min_attr_count, self.max_attr_count)
        elif isinstance(attr, attributes.StartActivityMinorAttribute):
            df = self.start_activity_PQL(self.min_attr_count, self.max_attr_count)
        elif isinstance(attr, attributes.ActivityTableColumnMinorAttribute):
            df_dyn_cat = self.aggregate_dynamic_categorical_PQL(
                self.min_attr_count, self.max_attr_count
            )
            df_dyn_num = self.aggregate_dynamic_numerical_PQL(
                utils.make_list(attr.aggregations)
            )
            df = PQL()
            df.add(df_dyn_cat)
            df.add(df_dyn_num)
            # df = utils.join_dfs([df_dyn_cat, df_dyn_num], keys=["caseid"] * 2)
        elif isinstance(attr, attributes.CaseTableColumnMinorAttribute):
            df_stat_cat = self.aggregate_static_categorical_PQL(
                self.min_attr_count, self.max_attr_count
            )
            df_stat_num = self.get_static_numerical_PQL()
            df = PQL()
            df.add(df_stat_cat)
            df.add(df_stat_num)
            # df = utils.join_dfs([df_stat_cat, df_stat_num], keys=["caseid"] * 2)
        else:
            raise NotAValidAttributeError(attr)

        return df
