from typing import List, Optional, Callable, Union, Dict, Tuple, Sequence, Any
import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
from dataclasses import dataclass, field
from pycelonis.celonis_api.pql.pql import PQL, PQLColumn, PQLFilter
from typing import Optional
from pycelonis.celonis_api.process_analytics.analysis import Analysis
import json
from enum import Enum


class MajorAttribute(Enum):
    ACTIVITY = "Activity"
    CASE = "Case"
    ordering = [ACTIVITY, CASE]
    def __lt__(self, other):
        return self.value <= other.value


class AttributeDataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    def __lt__(self, other):
        return self.value <= other.value


@dataclass(order=True)
class Attribute:
    major_attribute_type: MajorAttribute
    minor_attribute_type: str
    attribute_data_type: AttributeDataType
    df_attribute_name: str
    display_name: str
    correlation: Optional[float] = 0.
    p_val: Optional[float] = 1.
    unit: Optional[str] = ""
    label_influence: Optional[float] = None  # for categorical attributes only
    cases_with_attribute: Optional[int] = None  # for categorical attributes only


class EmptyTable:
    def __init__(self):
        self.columns = []

    def __bool__(self):
        return False


def get_aggregation_display_name(agg):
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

class Preprocessor:
    def __init__(
            self,
            datamodel,
            aggregations_dyn_cat=None,
            aggregations_dyn_num=None,
            gap=1,
            min_prefixes=1,
            max_prefixes=20,
            min_occurrences=10,
            chunksize=10000,
    ):

        self.dm = datamodel
        if aggregations_dyn_cat is None:
            self.aggregations_dyn_cat = ["last", "sum"]
        else:
            self.aggregations_dyn_cat = aggregations_dyn_cat

        if aggregations_dyn_num is None:
            self.aggregations_dyn_num = ["last", "sum"]
        else:
            self.aggregations_dyn_num = aggregations_dyn_num

        self.categorical_types = ["STRING", "BOOLEAN"]
        self.numerical_types = ["INTEGER", "FLOAT"]
        self.gap = gap
        self.min_prefixes = min_prefixes
        self.max_prefixes = max_prefixes
        self.min_occurrences = min_occurrences
        self.chunksize = chunksize
        # self.activity_df = None
        # self.case_df = None
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

    def _init_datamodel(self, dm):
        """Initialize datamodel parameters

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

            self.activity_case_key = foreign_key_case_id["columns"][0]["targetColumnName"]

            self.case_case_key = foreign_key_case_id["columns"][0]["sourceColumnName"]
            self.case_table_name = self.case_table.name
            self._set_dynamic_features_PQL()
            self._set_static_features_PQL()
        else:
            self.case_table = EmptyTable()
            self.case_case_key = ''
            self.case_table_name = ''
            self.activity_case_key = dm.data["processConfigurations"][0]['caseIdColumn']
            self._set_dynamic_features_PQL()
        # self.activity_df = activity_table.get_data_frame(chunksize=self.chunksize)
        # self.case_df = case_table.get_data_frame(chunksize=chunksize)

    def _set_static_features_PQL(self):
        for attribute in self.case_table.columns:
            if attribute['type'] in self.categorical_types and attribute['name'] not in [self.case_case_key,
                                                                                         self.sort_col]:
                self.static_categorical_cols.append(attribute['name'])
            elif attribute['type'] in self.numerical_types and attribute['name'] not in [self.case_case_key,
                                                                                         self.sort_col]:
                self.static_numerical_cols.append(attribute['name'])

    def _set_dynamic_features_PQL(self):
        for attribute in self.activity_table.columns:
            if attribute['type'] in self.categorical_types and attribute['name'] not in [self.activity_case_key,
                                                                                         self.sort_col]:
                self.dynamic_categorical_cols.append(attribute['name'])
            elif attribute['type'] in self.numerical_types and attribute['name'] not in [self.activity_case_key,
                                                                                         self.sort_col]:
                self.dynamic_numerical_cols.append(attribute['name'])

    def _adjust_string_values(self, l: List[str]):
        list_name = [el.replace('"', '\\"') for el in l]
        list_val = [el.replace("'", "\\'") for el in l]

        return list_val, list_name

    def compute_metrics(self, df, metrics=None):
        if metrics is None:
            metrics = ['influence', 'case_count', 'correlation']
        for attr in self.attributes:
            if attr.attribute_data_type == AttributeDataType.NUMERICAL:
                continue
            if 'influence' in metrics:
                label_val_0 = df[df[attr.df_attribute_name] == 0][self.label.df_attribute_name].mean()
                label_val_1 = df[df[attr.df_attribute_name] == 1][self.label.df_attribute_name].mean()
                attr.label_influence = label_val_1 - label_val_0
            if 'case_count' in metrics:
                attr.case_count = len(df[df[attr.df_attribute_name] == 1].index)
            if 'correlation' in metrics:
                label_series = df[self.label.df_attribute_name]
                attribute_df = df.drop(self.label.df_attribute_name, axis=1)
                correlations = attribute_df.corrwith(label_series)
                attr.correlation = correlations[attr.df_attribute_name]

    def one_hot_encoding_PQL(self, table: str, case_id, attributes, major_attribute:  MajorAttribute,
                             minor_attribute, min_vals=1, suffix: str = "", prefix: str=""):
        if suffix != "":
            suffix = " " + suffix
        if prefix != "":
            prefix = " " + prefix
        query = PQL()
        query.add(PQLColumn(name="caseid", query="\"" + table + "\".\"" + case_id + "\""))
        for attribute in attributes:
            query_unique = PQL()
            query_unique.add(PQLColumn(name="values", query="DISTINCT(\"" + table + "\".\"" + attribute + "\")"))
            query_unique.add(PQLColumn(name="count", query="COUNT_TABLE(\"" + self.case_table_name + "\")"))

            df_unique_vals = self.dm.get_data_frame(query_unique)
            # remove too few counts
            df_unique_vals = df_unique_vals[df_unique_vals['count'] >= min_vals]
            unique_values = list(df_unique_vals["values"])
            # Remove None values
            unique_values = [x for x in unique_values if x is not None]

            # Add escaping characters
            unique_vals_val, unique_vals_name = self._adjust_string_values(unique_values)
            for val_val, val_name in zip(unique_vals_val, unique_vals_name):
                df_attr_name = prefix + table + "_" + attribute + "_" + val_name + suffix
                display_name = prefix + table + "." + attribute + " = " + val_name + suffix

                major_attribute_type = major_attribute
                minor_attribute_type = minor_attribute

                attr_obj = Attribute(major_attribute_type, minor_attribute_type, AttributeDataType.CATEGORICAL,
                                     df_attr_name, display_name)
                self.attributes.append(attr_obj)
                self.attributes_dict[df_attr_name] = attr_obj

                query.add(PQLColumn(name=df_attr_name,
                                    query="SUM(CASE WHEN \"" + table + "\".\"" + attribute + "\" = '" + val_val + "' THEN 1 ELSE 0 END)"))
        dataframe = self.dm.get_data_frame(query)
        return dataframe

    def _aggregate_static_categorical_PQL(self, min_vals: int):
        major_attribute = MajorAttribute.CASE
        minor_attribute = "Case Table column"
        df_static_categorical = self.one_hot_encoding_PQL(self.case_table_name, self.case_case_key,
                                                          self.static_categorical_cols, major_attribute,
                                                          minor_attribute, min_vals=min_vals)
        # df_static_categorical = self.one_hot_encoding_PQL(self.case_table_name, self.case_case_key, self.static_categorical_cols)
        # Remove values with too few occurences per case key, can this be done in PQL directly???
        #df_static_categorical = df_static_categorical.loc[:, (df_static_categorical[df_static_categorical.drop(
        # "caseid",  axis=1) > 0].count( axis=0) >= min_vals) | (df_static_categorical.columns == "caseid")]
        df_static_categorical = self._conv_dtypes_PQL(df_static_categorical, ["object"], "category")
        return df_static_categorical

    def _aggregate_static_numerical_PQL(self):
        query = PQL()
        query.add(PQLColumn(name="caseid", query="\"" + self.case_table_name + "\".\"" + self.case_case_key + "\""))
        for attribute in self.static_numerical_cols:
            df_attr_name = self.case_table_name + "_" + attribute
            display_name = self.case_table_name + "." + attribute
            attr_obj = Attribute(MajorAttribute.ACTIVITY, "Case Table column", AttributeDataType.NUMERICAL,
                                 df_attr_name, display_name)
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj
            query.add(PQLColumn(name=df_attr_name,
                                query="\"" + self.case_table_name + "\"." + "\"" + attribute + "\""))
        dataframe = self.dm.get_data_frame(query)
        return dataframe

    def _aggregate_dynamic_categorical_PQL(self, min_vals: int=1):
        major_attribute = MajorAttribute.ACTIVITY
        minor_attribute = "Activity Table column"
        df_dynamic_categorical = self.one_hot_encoding_PQL(self.activity_table_name, self.activity_case_key,
                                                           self.dynamic_categorical_cols, major_attribute,
                                                           minor_attribute, min_vals=min_vals)
        # Remove values with too few occurences per case key, can this be done in PQL directly???
        #df_dynamic_categorical = df_dynamic_categorical.loc[:, (df_dynamic_categorical[
        #                                                            df_dynamic_categorical.drop("caseid",
        #                                                                                        axis=1) > 0].count(
        #    axis=0) >= min_vals) | (df_dynamic_categorical.columns == "caseid")]
        df_dynamic_categorical = self._conv_dtypes_PQL(df_dynamic_categorical, ["object"], "category")
        return df_dynamic_categorical

    def _aggregate_dynamic_numerical_PQL(self, aggregations = None):
        if aggregations is None:
            aggregations = ['AVG']
        query = PQL()
        query.add(PQLColumn(name="caseid",
                            query="DISTINCT( \"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\")"))
        for agg in aggregations:
            for attribute in self.dynamic_numerical_cols:
                df_attr_name = self.activity_table_name + "_" + agg + "_" + attribute
                display_name = self.activity_table_name + "." + attribute + " (" + get_aggregation_display_name(
                    agg) + ")"
                attr_obj = Attribute(MajorAttribute.ACTIVITY, "Activity Table column", AttributeDataType.NUMERICAL,
                                     df_attr_name, display_name)
                self.attributes.append(attr_obj)
                self.attributes_dict[df_attr_name] = attr_obj
                query.add(PQLColumn(name=df_attr_name,
                                    query= agg + "(\"" + self.activity_table_name + "\"." + "\"" + attribute + "\")"))
        dataframe = self.dm.get_data_frame(query)

        return dataframe

    def _aggregate_dynamic_categorical_prefixes_PQL(self, max_prefixes, min_vals):
        # one-hot-encoding
        dataframe = self.one_hot_encoding_PQL(self.activity_table_name, self.activity_case_key,
                                              self.dynamic_categorical_cols)

        # remove too rare values
        dataframe_enough_vals = dataframe.loc[:,
                                (dataframe[dataframe.drop("caseid", axis=1) > 0].count(axis=0) >= min_vals) | (
                                        dataframe.columns == "caseid")]
        query = PQL()
        query.add(
            PQLColumn(name="caseid", query="\"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\""))
        # Running total
        is_empty = True
        for attribute in self.dynamic_categorical_cols:
            query_unique = PQL()
            query_unique.add(
                PQLColumn(name="values", query="DISTINCT(\"" + self.activity_table_name + "\".\"" + attribute + "\")"))
            unique_values = list(self.dm.get_data_frame(query_unique)["values"])
            # Remove None values
            unique_values = [x for x in unique_values if x is not None]
            if unique_values:
                is_empty = False
            # Add escaping characters
            unique_vals_val, unique_vals_name = self._adjust_string_values(unique_values)
            for val_val, val_name in zip(unique_vals_val, unique_vals_name):
                col_name = "@@@" + self.activity_table_name + "@@" + attribute + "@" + val_name
                if col_name in dataframe_enough_vals.columns:
                    query.add(
                        PQLColumn(name="@@@@" + self.activity_table_name + "@@@" + attribute + "@@LAST@" + val_name,
                                  query="CASE WHEN \"" + self.activity_table_name + "\".\"" + attribute + "\" = '" + val_val + "' THEN 1 ELSE 0 END"))
                    query.add(
                        PQLColumn(name="@@@@" + self.activity_table_name + "@@@" + attribute + "@@SUM@" + val_name,
                                  query="RUNNING_SUM(CASE WHEN \"" + self.activity_table_name + "\".\"" + attribute + "\" = '" +
                                        val_val + "' THEN 1 ELSE 0 END, PARTITION BY( \"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\"))"))
        if is_empty:
            return None
        dataframe = self.dm.get_data_frame(query)
        dataframe = self._conv_dtypes_PQL(dataframe, ["object"], "category")

        df_prefixes = self.extract_prefixes_categorical_PQL(dataframe, "caseid", max_prefixes)

        return df_prefixes

    def _aggregate_dynamic_numerical_prefixes_PQL(self, max_prefixes):
        if not self.dynamic_numerical_cols:
            return None

        query = PQL()
        query.add(PQLColumn(name="caseid",
                            query="DISTINCT( \"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\")"))
        for attribute in self.dynamic_numerical_cols:
            query.add(PQLColumn(name="@@@" + self.activity_table_name + "@@mean@" + attribute,
                                query="RUNNING_SUM(\"" + self.activity_table_name + "\"." + "\"" + attribute + "\", PARTITION BY( \"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\"))"))
        dataframe = self.dm.get_data_frame(query)
        df_prefixes = self.extract_prefixes_numerical_PQL(dataframe, "caseid", max_prefixes)

        return df_prefixes

    def extract_prefixes_numerical_PQL(self, df, case_key, max_prefixes):
        # Currently only average supported
        df = df.copy()
        df["case_length"] = df.groupby(case_key, observed=True)[case_key].transform(len)

        df_prefixes = pd.DataFrame(
            columns=df.columns.tolist() + [self.prefixes_case_key, "num_activities"]
        )
        for i in range(max_prefixes):
            g = df.groupby(case_key, observed=True)
            tmp = df[g.cumcount() == i]
            columns_numeric = tmp.columns.tolist()
            columns_numeric = [el for el in columns_numeric if el not in [case_key, "case_length"]]
            tmp[columns_numeric] = tmp[columns_numeric] / (i + 1)
            tmp[self.prefixes_case_key] = tmp[case_key].apply(
                lambda x: f"{x}_{i + 1}"
            )
            tmp["num_activities"] = i + 1
            # Add 'num_prefixes' to self.static_numerical_cols
            df_prefixes = pd.concat([df_prefixes, tmp], axis=0)
        df_prefixes = df_prefixes.drop("case_length", axis=1)
        return df_prefixes

    def extract_prefixes_categorical_PQL(self, df, case_key, max_prefixes):
        df = df.copy()
        df["case_length"] = df.groupby(case_key, observed=True)[case_key].transform(len)

        df_prefixes = pd.DataFrame(
            columns=df.columns.tolist() + [self.prefixes_case_key, "num_activities"]
        )
        for i in range(max_prefixes):
            g = df.groupby(case_key, observed=True)
            tmp = df[g.cumcount() == i]
            tmp[self.prefixes_case_key] = tmp[case_key].apply(
                lambda x: f"{x}_{i + 1}"
            )
            tmp["num_activities"] = i + 1
            # Add 'num_prefixes' to self.static_numerical_cols
            df_prefixes = pd.concat([df_prefixes, tmp], axis=0)
        df_prefixes = df_prefixes.drop("case_length", axis=1)
        return df_prefixes

    def _extract_prefixes_remaining_time_PQL(self, df, max_prefixes):
        df["case_length"] = df.groupby("caseid", observed=True)["caseid"].transform(len)
        df_prefixes = pd.DataFrame(
            columns=df.columns.tolist() + [self.prefixes_case_key]
        )
        for i in range(max_prefixes):
            g = df.groupby("caseid", observed=True)
            tmp = df[g.cumcount() == i]
            tmp[self.prefixes_case_key] = tmp["caseid"].apply(
                lambda x: f"{x}_{i + 1}"
            )
            df_prefixes = pd.concat([df_prefixes, tmp], axis=0)
        df_prefixes = df_prefixes.drop("case_length", axis=1)
        return df_prefixes

    def remaining_time_PQL(self, max_prefixes):

        query = PQL()
        query.add(PQLColumn(name="caseid",
                            query="TARGET(\"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\")"))
        query.add(PQLColumn(name="@@source",
                            query="SOURCE(\"" + self.activity_table_name + "\".\"" + self.activity_col + "\", ANY_OCCURRENCE [ ] TO LAST_OCCURRENCE [ ])"))
        query.add(PQLColumn(name="@@target",
                            query="TARGET(\"" + self.activity_table_name + "\".\"" + self.activity_col + "\")"))
        # query.add(PQLColumn(name="timestamp", query="\"" + activity_table + "\".\"" + timestamp_key + "\""))
        query.add(PQLColumn(name="@@time_to_end",
                            query="SECONDS_BETWEEN(SOURCE(\"" + self.activity_table_name + "\".\"" + self.eventtime_col + "\"), TARGET(\"" + self.activity_table_name + "\".\"" + self.eventtime_col + "\"))"))
        dataframe = self.dm.get_data_frame(query)
        dataframe = dataframe.drop(['@@source', '@@target'], axis=1)
        dataframe = self._extract_prefixes_remaining_time_PQL(dataframe, max_prefixes)

        return dataframe

    def get_query_case_ids(self):
        return PQLColumn(name="caseid",
                         query="DISTINCT( \"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\")")

    def one_hot_encode_special(self, min_vals, query_str, attribute_name, major_attribute: MajorAttribute,
                               minor_attribute: str):
        """ One hot encoding with a special query.
        query is string with what comes within the DISTINCT() brackets in the frist query and then in the CASE WHEN in the second query


        :param min_vals: minimum values
        :param query_str: the query
        :param attribute_name: the attribute name in "attribute_name = value"
        :param major_attribute: the major attribute
        :param minor_attribute: the minor attribute
        :return:
        """

        query_unique = PQL()
        query_unique.add(PQLColumn(name="values", query="DISTINCT(" + query_str + ")"))
        query_unique.add(PQLColumn(name="count", query="COUNT_TABLE(\"" + self.case_table_name + "\")"))

        df_unique_vals = self.dm.get_data_frame(query_unique)
        # remove too few counts
        df_unique_vals = df_unique_vals[df_unique_vals['count'] >= min_vals]
        unique_values = list(df_unique_vals["values"])
        # Remove None values
        unique_values = [x for x in unique_values if x is not None]
        # Add escaping characters
        unique_vals_val, unique_vals_name = self._adjust_string_values(unique_values)
        query = PQL()
        query.add(self.get_query_case_ids())
        for val_val, val_name in zip(unique_vals_val, unique_vals_name):
            df_attr_name = attribute_name + " = " + val_name
            display_name = attribute_name + " = " + val_val
            attr_obj = Attribute(MajorAttribute.ACTIVITY, minor_attribute, AttributeDataType.NUMERICAL,
                                 df_attr_name, display_name)
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj
            query.add(PQLColumn(name=df_attr_name,
                                query="SUM(CASE WHEN " + query_str + " = " + "'" + val_val + "' THEN 1 ELSE 0 END)"))
        dataframe = self.dm.get_data_frame(query)
        return dataframe

    def start_activity_PQL(self, min_vals):
        attribute_name = "Start activity"
        major_attribute = MajorAttribute.ACTIVITY
        minor_attribute = attribute_name
        query_str = "PU_FIRST(\"" + self.case_table_name + "\", \"" + self.activity_table_name + "\".\"" + self.activity_col + "\")"
        df = self.one_hot_encode_special(min_vals, query_str, attribute_name, major_attribute, minor_attribute)

        return df

    def end_activity_PQL(self, min_vals):
        attribute_name = "End activity"
        major_attribute = MajorAttribute.ACTIVITY
        minor_attribute = attribute_name
        query_str = "PU_LAST(\"" + self.case_table_name + "\", \"" + self.activity_table_name + "\".\"" + self.activity_col + "\")"
        df = self.one_hot_encode_special(min_vals, query_str, attribute_name, major_attribute, minor_attribute)
        return df

    def _binarize(self, x, th=1):
        """
        set all values larger than th to 1, else to 0
        x: Series

        """
        x[x > 1] = 1
        return x

    def binary_activity_occurence_PQL(self, min_vals):
        suffix = "(occurence)"
        major_attribute = MajorAttribute.ACTIVITY
        minor_attribute = "Activity occurence"
        df_activities = self.one_hot_encoding_PQL(self.activity_table_name, self.activity_case_key,
                                                  [self.activity_col], major_attribute, minor_attribute, min_vals,
                                                  suffix)
        # Remove values with too few occurences per case key, can this be done in PQL directly???
        df_activities[df_activities.drop('caseid', axis=1).columns] = df_activities[
            df_activities.drop('caseid', axis=1).columns].apply(lambda x: self._binarize(x, 0), axis=1)
        df_activities = self._conv_dtypes_PQL(df_activities, ["object"], "category")
        return df_activities

    def binary_rework_PQL(self, min_vals):
        suffix = "(rework)"
        major_attribute = MajorAttribute.ACTIVITY
        minor_attribute = "Rework"

        df_activities = self.one_hot_encoding_PQL(self.activity_table_name, self.activity_case_key,
                                                  [self.activity_col], major_attribute, minor_attribute, min_vals,
                                                  suffix)
        # Remove values with too few occurences per case key, can this be done in PQL directly???
        df_activities[df_activities.drop('caseid', axis=1).columns] = df_activities[
            df_activities.drop('caseid', axis=1).columns].apply(lambda x: self._binarize(x, 1), axis=1)
        df_activities = self._conv_dtypes_PQL(df_activities, ["object"], "category")
        return df_activities

    def num_events(self):
        df_attr_name = "Event count"
        display_name = "Event count"
        major_attribute = MajorAttribute.ACTIVITY
        minor_attribute = "Event count"
        attr_obj = Attribute(major_attribute, minor_attribute, AttributeDataType.NUMERICAL, df_attr_name,
                             display_name)
        self.attributes.append(attr_obj)
        self.attributes_dict[df_attr_name] = attr_obj
        q_num_events = "PU_COUNT(\"" + self.case_table_name + "\", \"" + self.activity_table_name + "\".\"" + self.activity_col + "\")"
        query = PQL()
        query.add(self.get_query_case_ids())
        query.add(PQLColumn(name='num_events', query=q_num_events))
        df = self.dm.get_data_frame(query)
        return df

    def work_in_progress_PQL(self, aggregations=None):
        if aggregations is None:
            aggregations = ['MIN', 'MAX', 'AVG']

        query = PQL()
        query.add(self.get_query_case_ids())

        for agg in aggregations:
            agg_display_name = get_aggregation_display_name(agg)
            df_attr_name = "Work in progress" + " (" + agg_display_name + ")"
            display_name = "Work in progress" + " (" + agg_display_name + ")"
            major_attribute = MajorAttribute.CASE
            minor_attribute = "Work in progress " + " (" + agg_display_name + ")"
            attr_obj = Attribute(major_attribute, minor_attribute, AttributeDataType.NUMERICAL, df_attr_name,
                                 display_name)
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj

            q = "PU_" + agg + " ( \"" + self.case_table_name + "\", RUNNING_SUM( CASE WHEN INDEX_ACTIVITY_ORDER ( \"" + self.activity_table_name + "\".\"" + self.activity_col + "\" ) = 1 THEN 1 WHEN INDEX_ACTIVITY_ORDER_REVERSE ( \"" + self.activity_table_name + "\".\"" + self.activity_col + "\" ) = 1 THEN -1 ELSE 0 END, ORDER BY ( \"" + self.activity_table_name + "\".\"" + self.eventtime_col + "\" ) ) )"
            query.add(PQLColumn(name=df_attr_name, query=q))
        df = self.dm.get_data_frame(query)
        return df

    def run_total_time_PQL(self, min_vals, time_aggregation="DAYS"):
        start_activity_df = self.start_activity_PQL(min_vals)
        end_activity_df = self.end_activity_PQL(min_vals)
        binary_activity_occurence_df = self.binary_activity_occurence_PQL(min_vals)
        binary_rework_df = self.binary_rework_PQL(min_vals)
        work_in_progress_df = self.work_in_progress_PQL(aggregations=['AVG'])

        static_cat_df = self._aggregate_static_categorical_PQL(min_vals)
        print(f"length of static_cat_df: {len(static_cat_df.index)}")
        static_num_df = self._aggregate_static_numerical_PQL()
        print(f"length of static_num_df: {len(static_num_df.index)}")
        dyn_cat_df = self._aggregate_dynamic_categorical_PQL(min_vals)
        print(f"length of dyn_cat_df: {len(dyn_cat_df.index)}")
        dyn_num_df = self._aggregate_dynamic_numerical_PQL()
        print(f"length of dyn_num_df: {len(dyn_num_df.index)}")
        total_time_df = self.total_time_PQL(time_aggregation, is_label=True)
        print(f"length of total_time_df: {len(total_time_df.index)}")
        joined_df = self._join_dfs(
            [start_activity_df, end_activity_df, binary_activity_occurence_df, binary_rework_df, work_in_progress_df,
             static_cat_df, static_num_df, dyn_cat_df, dyn_num_df, total_time_df], keys=['caseid'] * 10)
        self.compute_metrics(joined_df)
        return joined_df

    def total_time_PQL(self, time_aggregation, is_label: bool = False):

        df_attr_name = "case duration"
        display_name = "case duration"
        major_attribute = MajorAttribute.CASE
        minor_attribute = "case duration"
        attr_obj = Attribute(major_attribute, minor_attribute, AttributeDataType.NUMERICAL, df_attr_name,
                             display_name, time_aggregation.lower())
        if is_label:
            self.label = attr_obj
            self.label_dict[df_attr_name] = attr_obj
        else:
            self.attributes.append(attr_obj)
            self.attributes_dict[df_attr_name] = attr_obj

        query = PQL()
        query.add(PQLColumn(name="caseid", query="\"" + self.case_table_name + "\".\"" + self.case_case_key + "\""))
        q_total_time = (
                "(CALC_THROUGHPUT(ALL_OCCURRENCE['Process Start'] TO ALL_OCCURRENCE['Process End'], REMAP_TIMESTAMPS(\""
                + self.activity_table_name
                + '"."'
                + self.eventtime_col
                + '", '
                + time_aggregation
                + ")))"
        )
        query.add(PQLColumn(q_total_time, 'case duration'))
        dataframe = self.dm.get_data_frame(query)
        return dataframe

    def _extract_prefixes_past_time_PQL(self, df, max_prefixes):
        df["case_length"] = df.groupby("caseid", observed=True)["caseid"].transform(len)
        df_prefixes = pd.DataFrame(
            columns=df.columns.tolist() + [self.prefixes_case_key]
        )
        for i in range(max_prefixes):
            g = df.groupby("caseid", observed=True)
            tmp = df[g.cumcount() == i]
            tmp[self.prefixes_case_key] = tmp["caseid"].apply(
                lambda x: f"{x}_{i + 2}"
            )
            df_prefixes = pd.concat([df_prefixes, tmp], axis=0)
        df_prefixes = df_prefixes.drop("case_length", axis=1)
        return df_prefixes

    def past_time_PQL(self, max_prefixes):
        query = PQL()

        query = PQL()
        query.add(PQLColumn(name="caseid",
                            query="TARGET(\"" + self.activity_table_name + "\".\"" + self.activity_case_key + "\")"))
        query.add(PQLColumn(name="@@source",
                            query="SOURCE(\"" + self.activity_table_name + "\".\"" + self.activity_col + "\", FIRST_OCCURRENCE [ ] TO ANY_OCCURRENCE [ ])"))
        query.add(PQLColumn(name="@@target",
                            query="TARGET(\"" + self.activity_table_name + "\".\"" + self.activity_col + "\")"))
        # query.add(PQLColumn(name="timestamp", query="\"" + activity_table + "\".\"" + timestamp_key + "\""))
        query.add(PQLColumn(name="@@past_time",
                            query="SECONDS_BETWEEN(SOURCE(\"" + self.activity_table_name + "\".\"" + self.eventtime_col + "\"), TARGET(\"" + self.activity_table_name + "\".\"" + self.eventtime_col + "\"))"))
        dataframe = self.dm.get_data_frame(query)
        dataframe = dataframe.drop(['@@source', '@@target'], axis=1)
        dataframe = self._extract_prefixes_past_time_PQL(dataframe, max_prefixes)

        return dataframe

    def time_from_case_start(analysis, activity_table, case_id, attribute, timestamp_key):
        query = PQL()

        query.add(PQLColumn(name="values", query="DISTINCT(\"" + activity_table + "\".\"" + attribute + "\")"))

        unique_values = list(analysis.get_data_frame(query)["values"])

        query = PQL()
        query.add(PQLColumn(name="caseid", query="TARGET(\"" + activity_table + "\".\"" + case_id + "\")"))

        for val in unique_values:
            query.add(PQLColumn(name="@@min_time_from_start_" + attribute + "_" + val,
                                query="MIN(CASE WHEN TARGET(\"" + activity_table + "\".\"" + attribute + "\") = '" + val + "' THEN SECONDS_BETWEEN(SOURCE(\"" + activity_table + "\".\"" + timestamp_key + "\", FIRST_OCCURRENCE [ ] TO ANY_OCCURRENCE [ ]), TARGET(\"" + activity_table + "\".\"" + timestamp_key + "\")) ELSE NULL END)"))

        dataframe = analysis.get_data_frame(query)
        return dataframe

    def run_remaining_time_PQL(self, min_vals, max_prefixes):
        static_cat_df = self._aggregate_static_categorical_PQL(min_vals)
        static_num_df = self._aggregate_static_numerical_PQL()
        dyn_cat_df_prefixes = self._aggregate_dynamic_categorical_prefixes_PQL(max_prefixes, min_vals)
        dyn_num_df_prefixes = self._aggregate_dynamic_numerical_prefixes_PQL(max_prefixes)
        past_time_df = self.past_time_PQL(max_prefixes)
        remaining_time_df = self.remaining_time_PQL(max_prefixes)
        # join dfs
        # first with prefixes

        joined_df = self._join_dfs([dyn_cat_df_prefixes, dyn_num_df_prefixes, past_time_df, remaining_time_df],
                                   keys=[self.prefixes_case_key] * 4)

        joined_df = self._join_dfs([joined_df, static_cat_df, static_num_df], keys=["caseid"] * 3)

        joined_df["@@time_to_end"] = joined_df["@@time_to_end"].fillna(0)
        joined_df["@@past_time"] = joined_df["@@past_time"].fillna(0)

        return joined_df

    def _join_dfs(self, dfs: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
        """Perform a Left outer join on two DataFrame. Only the key of the first
        DataFrame is kept

        :param dfs: list of at least two DataFrames
        :param keys: columns to join on. But be of same length as dfs
        :return: joined DataFrame
        """
        df_result = None
        for i in range(0, len(dfs)):
            if dfs[i] is not None:
                df_result = dfs[i]
                break

        for i in range(1, len(dfs)):
            if dfs[i] is None:
                continue
            # Rmove common columns from one of those
            common_columns = np.intersect1d(df_result.columns, dfs[i].columns).tolist()
            if keys[i] in common_columns:
                common_columns.remove(keys[i])
            dfs[i] = dfs[i].drop(common_columns, axis=1)
            df_result = pd.merge(df_result, dfs[i], how='left', left_on=keys[0], right_on=keys[i])

            # Drop right key if it's different from the left key
            if keys[0] != keys[i]:
                df_result.drop(keys[i], axis=1)
        return df_result

    def save_config(self):
        pass

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

    def _gen_prefixes(
            self,
            df: pd.DataFrame,
            case_key: str,
            prefixes_case_key: str,
            min_prefixes: int = 1,
            max_prefixes: int = 20,
            gap: int = 1,
    ) -> pd.DataFrame:
        """Generate Prefixes. Columns that are added are: 'num_prefixes',
        <prefixes_case_key>

        :param df: input DataFrame
        :param case_key: case key
        :param prefixes_case_key: case key for prefixes
        :param min_prefixes: minimum prefixes
        :param max_prefixes: maximum prefixes
        :param gap: gap between prefixes
        :return: DataFrame with prefixes
        """
        df = df.copy()
        df["case_length"] = df.groupby(case_key, observed=True)[case_key].transform(len)

        # Create new DataFrame with columns from input df and column 'num_activities'
        # which has the number of prefixes.
        df_prefixes = pd.DataFrame(
            columns=df.columns.tolist() + [prefixes_case_key, "num_activities"]
        )
        for num_prefixes in range(min_prefixes, max_prefixes + 1, gap):
            tmp = (
                df[df["case_length"] >= num_prefixes]
                    .groupby(case_key, observed=True)
                    .head(num_prefixes)
            )
            tmp[prefixes_case_key] = tmp[case_key].apply(
                lambda x: f"{x}_{num_prefixes}"
            )
            tmp["num_activities"] = num_prefixes
            # Add 'num_prefixes' to self.static_numerical_cols
            df_prefixes = pd.concat([df_prefixes, tmp], axis=0)
        self.static_numerical_cols.append("num_activities")
        return df_prefixes

    def _remove_few_occurrences(
            self, df: pd.DataFrame, columns: List[str], min_occurrences: int = 20
    ):
        """Set values that occur to few times to np.nan

        :param df: input DataFrame
        :param columns: columns from which to remove values with few occurences
        :param min_occurrences: minimum occurrences
        :return: dataframe with removed few occurrences
        """
        df = df.copy()
        # Remove few occurances
        for col_name in columns:
            counts = df[col_name].value_counts()
            repl = counts[counts < min_occurrences].index.tolist()
            df[col_name].fillna(np.nan, inplace=True)
            df[col_name].replace(repl, np.nan, inplace=True)
            df[col_name] = df[col_name].cat.remove_unused_categories()

        return df

    def _remove_few_occurrences_inference(self, df: pd.DataFrame, values_dict: Dict[str, Sequence[str]]):
        """ Remove the (categorical) values that are not in values_dict.

        :param df: input DataFrame
        :param values_dict: dictionary that contains the values for the categorical columns.
        :return: DataFrame with removed few occurences
        """
        df = df.copy()
        # Remove occurences that are not in values_dict
        for col, vals in values_dict.items():

            if col in df.columns:
                df.loc[~df[col].isin(vals), col] = np.nan
                df[col] = df[col].cat.remove_unused_categories()

        return df

    def _add_missing_columns(self, df: pd.DataFrame, cols: Sequence[str]):
        """ Add columns from col into the DataFrame if they are not in it yet.

        :param df: input DataFrame
        :param cols: column names
        :return: the DataFrame with the added columns
        """
        for col in cols:
            if col not in df.columns:
                df[col] = 0
        return df

    def _save_categorical_values(self, activity_df: Optional[pd.DataFrame], case_df: Optional[pd.DataFrame]):
        """ Save the categorical values in the corresponding dicts.

        :param activity_df: activits DataFrame
        :param case_df: case DataFrame
        :return:
        """

        if activity_df is not None:
            for col in self.dynamic_categorical_cols:
                unique_vals = activity_df[col].dropna().unique()
                self.dynamic_categorical_values[col] = unique_vals.tolist()

        if case_df is not None:
            for col in self.static_categorical_cols:
                unique_vals = case_df[col].dropna().unique()
                self.static_categorical_values[col] = unique_vals.tolist()

    def _remove_datetime(
            self, df: pd.DataFrame, exclude: Optional[str] = None
    ) -> pd.DataFrame:
        """Remove datetime64 columns excluding the column specified 'exclude'

        :param df: input DataFrame
        :param exclude: datetime column to exclude from being removed
        :return: DataFrame with removed datetime columns
        """
        datetime_cols = df.select_dtypes("datetime").columns.tolist()
        if exclude is not None and exclude in datetime_cols:
            datetime_cols.remove(exclude)

        if datetime_cols:
            df = df.copy()
            df = df.drop(datetime_cols, axis=1)

        return df

    def _conv_dtypes(
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

    def _aggregate_static_categorical(
            self, df: pd.DataFrame, case_key: str, columns: List[str]
    ):
        """Basically one-hot encoding of static categorical features (from case table)

        :param df: input DataFrame
        :param case_key: the case key column, usually the prefix-case-key columns
        :param columns: the columns to which to apply the aggregations
        :return: DataFrame with aggregations
        """
        if not columns:
            df_agg = pd.DataFrame(df.groupby(case_key, observed=True)[case_key].agg("last"))
            df_agg.index.name = None
            return df_agg

        df_relevant = df[[case_key] + columns]
        df_dummies_rel = pd.get_dummies(
            df_relevant, prefix=columns, prefix_sep=" = ", columns=columns, sparse=True
        )
        df_dummies_rel = df_dummies_rel.groupby(case_key, observed=True).agg("last").reset_index()
        return df_dummies_rel

    def _aggregate_static_numerical(
            self, df: pd.DataFrame, case_key: str, columns: List[str]
    ) -> pd.DataFrame:
        """Return the static numerical features (from case table) from the input
        DataFrame
        :param df: input DataFrame
        :param case_key: the case key column, usually the prefix-case-key columns
        :param columns: the columns to which to apply the aggregations
        :return: DataFrame with aggregations
        """
        if not columns:
            df_agg = pd.DataFrame(df.groupby(case_key, observed=True)[case_key].agg("last"))
            df_agg.index.name = None
            return df_agg

        df_relevant = df[[case_key] + columns]
        df_relevant = df_relevant.groupby(case_key, observed=True).agg("last").reset_index()
        return df_relevant

    def _aggregate_dynamic_categorical(
            self,
            df: pd.DataFrame,
            case_key: str,
            columns: List[str],
            aggregations: List[str] = ["last", "sum"],
    ) -> pd.DataFrame:
        """Aggregate dynamic categorical columns. For this, one-hot-encoding is applied
        :param df: input DataFrame
        :param case_key: the case key column, usually the prefix-case-key columns
        :param columns: the columns to which to apply the aggregations
        :param aggregations: list with the aggregations to apply. The following
        aggregations are possible: 'last' and 'sum'
        :return: DataFrame with aggregations
        """

        if not columns:
            df_agg = pd.DataFrame(df.groupby(case_key, observed=True)[case_key].agg("last"))
            df_agg.index.name = None
            return df_agg

        df_relevant = df[[case_key] + columns]

        df_dummies_rel = pd.get_dummies(
            df_relevant, prefix=columns, prefix_sep=" = ", columns=columns, sparse=True
        )
        col_names = df_dummies_rel.columns
        df_agg = df_dummies_rel.groupby(case_key, observed=True).agg(aggregations).reset_index()
        df_agg.columns = [
            "_".join(col).strip() if col != (case_key, "") else col[0]
            for col in df_agg.columns.values
        ]

        return df_agg

    def _aggregate_dynamic_numerical(
            self,
            df: pd.DataFrame,
            case_key: str,
            columns: List[str],
            aggregations: Dict[str, Union[Callable, str]] = {
                "last": "last",
                "sum": np.sum,
                "mean": np.mean,
                "std": np.std,
                "min": np.min,
                "max": np.max,
            },
    ) -> pd.DataFrame:
        """Use aggregations on numerical columns

        :param df: input DataFrame
        :param case_key: the case key column, usually the prefix-case-key columns
        :param columns: the columns to which to apply the aggregations
        :param aggregations: dict containing a string as key that will be used for
        the column name and a function or
        string as value that can be used with pandas' DataFrameGroupBy.agg function.
        :return: DataFrame with aggregations
        """
        if not columns:
            df_agg = pd.DataFrame(df.groupby(case_key, observed=True)[case_key].agg("last"))
            df_agg.index.name = None
            return df_agg

        df_relevant = df[[case_key] + columns]
        df_agg = df_relevant.groupby(case_key, observed=True).agg(aggregations).reset_index()
        df_agg.columns = [
            "_".join(col).strip() if col != (case_key, "") else col[0]
            for col in df_agg.columns.values
        ]

        df_agg.fillna(0, inplace=True)
        return df_agg

    def _compute_past_time(
            self,
            df: pd.DataFrame,
            case_key: str,
            eventtime_col: str,
            past_time_col: Optional[str] = None,
    ):
        """Compute the past time from the start of the case till the end of the case.
        The time is returned in minutes

        :param df: input DataFrame
        :param case_key: name of the case key column (usually the prefix case_key
        column)
        :param time_col: name of the eventtime column
        :param past_time_col: name to use for the column that has the past time. If
        None, it will be <eventtime_col> + "_PAST[minutes]"
        :return: DataFrame with past time column
        """

        df_past_time = (
                (
                        df.groupby(case_key, observed=True)[eventtime_col].last()
                        - df.groupby(case_key, observed=True)[eventtime_col].first()
                )
                / pd.Timedelta("1 minute")
        ).reset_index()

        past_time_col_name = (
            past_time_col
            if past_time_col is not None
            else eventtime_col + "_PAST[minutes]"
        )
        df_past_time.rename(columns={eventtime_col: past_time_col_name}, inplace=True)

        return df_past_time

    def _get_cols_by_type(
            self, df: pd.DataFrame, dtypes: List[str], exclude: Optional[List[str]] = None
    ) -> List[str]:
        """Get the column names of a DataFrame by dtype

        :param df: input DataFrame
        :param dtypes: dtypes of column names
        :param exclude: List of column names to exclude
        :return: List with the colun names of the specified dtypes
        """
        if exclude is None:
            exclude = []
        cols = df.select_dtypes(dtypes)
        cols = list(set(cols) - set(exclude))
        return cols

    def _rename_col_names(self, activity_df: Optional[pd.DataFrame], case_df: Optional[pd.DataFrame]):
        """Adds prefix "<activity table name>_" to activity table columns and
        "<case table name>_" to case table columns. Also renames the relevant member
        variables.

        :param activity_df: activity DataFrame
        :param case_df: case DataFrame
        """

        # Columns to not rename
        cols_not_rename = [
            self.case_case_key,
            self.activity_case_key,
            self.eventtime_col,
            self.sort_col,
            self.activity_col,
        ]
        if activity_df is not None:
            activity_df.columns = [
                self.activity_table_name + "_" + col if col not in cols_not_rename else col
                for col in activity_df.columns.values
            ]
        if case_df is not None:
            case_df.columns = [
                self.case_table_name + "_" + col if col not in cols_not_rename else col
                for col in case_df.columns.values
            ]

        # self.case_case_key = self.case_table_name + "_" + self.case_case_key
        # self.activity_case_key = self.activity_table_name + "_" + self.activity_case_key
        # self.activity_col = self.activity_table_name + "_" + self.activity_col
        # self.eventtime_col = self.activity_table_name + "_" + self.eventtime_col
        # self.sort_col = self.activity_table_name + "_" + self.sort_col

    def _gen_label_future_activity(
            self, activity_df: pd.DataFrame, prefixes_df: pd.DataFrame, activity_name: str
    ) -> pd.DataFrame:
        """Generates the labels for the use case if an activity happens in the future.

        :param activity_df: the preprocessed activities DataFrame
        :param prefixes_df: the prefixes DataFrame
        :param activity_name: the name of the activity to check if it happens in the
        future
        :return: DataFrame with columns containing self.prefixes_case_key and the
        corresponding labels
        """

        # Set label column
        self.label_col = "label_future_" + activity_name

        label_activities_df = pd.DataFrame(activity_df[self.activity_case_key])

        # get number of appearences of activity for each case in activity_df
        col_name_appearances = activity_name + "_appearances"
        label_activities_df.loc[
            activity_df[self.activity_col] == activity_name, col_name_appearances
        ] = 1
        label_activities_df.loc[
            activity_df[self.activity_col] != activity_name, col_name_appearances
        ] = 0
        label_activities_df = label_activities_df.groupby(
            self.activity_case_key, observed=True, as_index=False
        )[col_name_appearances].sum()
        # add prefix_case_key
        label_activities_df = self._join_dfs(
            [
                label_activities_df,
                prefixes_df[[self.activity_case_key, self.prefixes_case_key]],
            ],
            [self.activity_case_key, self.activity_case_key],
        )
        label_activities_df = label_activities_df.drop_duplicates()

        # Accumulate number of appearences of activity for each prefic_case in
        # activity_df
        label_prefixes_df = prefixes_df[
            [self.activity_case_key, self.prefixes_case_key]
        ]
        label_prefixes_df.loc[
            prefixes_df[self.activity_col] == activity_name, col_name_appearances
        ] = 1
        label_prefixes_df.loc[
            prefixes_df[self.activity_col] != activity_name, col_name_appearances
        ] = 0
        label_prefixes_df = label_prefixes_df.groupby(
            self.prefixes_case_key, observed=True, as_index=False
        ).agg({col_name_appearances: "sum", self.activity_case_key: "first"})

        # Sort DataFrames by prefixes_case_key
        label_activities_df = label_activities_df.sort_values(
            by=[self.prefixes_case_key]
        )
        label_prefixes_df = label_prefixes_df.sort_values(by=[self.prefixes_case_key])

        # Get labels where col_name_appearances is smaller in prefixes df than in
        # activity df
        labels = (
                label_prefixes_df[col_name_appearances].reset_index(drop=True)
                < label_activities_df[col_name_appearances].reset_index(drop=True)
        ).astype(int)

        labels_df = pd.DataFrame(label_prefixes_df[self.prefixes_case_key])
        labels_df[self.label_col] = labels

        return labels_df

    def _gen_label_remaining_execution_time(
            self, activity_df: pd.DataFrame, prefixes_df: pd.DataFrame
    ):
        """Generate lables for remaining execution time.

        :param activity_df: the preprocessed activities DataFrame
        :param prefixes_df: the prefixes DataFrame
        :return: DataFrame with columns containing self.prefixes_case_key and the
        corresponding labels
        """
        self.label_col = "label_remaining_time"
        total_eventtime_df = self._compute_past_time(
            activity_df,
            self.activity_case_key,
            eventtime_col=self.eventtime_col,
            past_time_col="total_time",
        )
        # add prefix_case_key
        total_eventtime_df = self._join_dfs(
            [
                total_eventtime_df,
                prefixes_df[[self.activity_case_key, self.prefixes_case_key]],
            ],
            [self.activity_case_key, self.activity_case_key],
        )
        total_eventtime_df = total_eventtime_df.drop_duplicates()

        past_eventtime_df = self._compute_past_time(
            prefixes_df,
            self.prefixes_case_key,
            eventtime_col=self.eventtime_col,
            past_time_col="past_time",
        )

        # Sort DataFrames by prefixes_case_key
        total_eventtime_df = total_eventtime_df.sort_values(
            by=[self.prefixes_case_key]
        )
        past_eventtime_df = past_eventtime_df.sort_values(
            by=[self.prefixes_case_key]
        )

        # compute the remaining case time
        remaining_time = total_eventtime_df["total_time"].reset_index(
            drop=True
        ) - past_eventtime_df["past_time"].reset_index(drop=True)

        labels_df = pd.DataFrame(past_eventtime_df[self.prefixes_case_key])
        labels_df[self.label_col] = remaining_time

        return labels_df

    def _gen_label_total_time(self, df: pd.DataFrame, case_key: str) -> pd.DataFrame:
        """ Gererate label for total trace time

        :param df: imput DataFrame
        :param case_key: the case key to group on
        :return: DataFrame with the added label
        """
        self.label_col = "label_TOTAL_TIME"

        return self._compute_past_time(df, case_key, self.eventtime_col, self.label_col)

    def _gen_label_transition_times(
            self, prefixes_df: pd.DataFrame, activity_out: Sequence[str], activity_in: Sequence[str]
    ) -> pd.DataFrame:
        """Generate lables for transition times (The time from the 2nd last activity
        to the last activity).

        :param prefixes_df: the prefixes DataFrame
        :param activity_out: outgoing activity
        :param activity_in: incoming_activity
        :return: DataFrame with prefix_case_key and transition_time of last transition.
        """
        min_size = prefixes_df.groupby(self.prefixes_case_key, observed=True).size().min()

        # print(prefixes_df)
        # def pr(g):
        #    #if len(g.index) == 1:
        #    print(g)
        # prefixes_df.groupby(self.prefixes_case_key, as_index=False).apply(pr)
        # print(min_size)
        def trans_time(g):
            if len(g.index) < 2:
                return
            else:
                return (g[self.eventtime_col].iloc[-1] - g[self.eventtime_col].iloc[-2]) / pd.Timedelta("1 minute")

        # labels_df = prefixes_df.groupby(self.prefixes_case_key, as_index=False)[
        #    self.eventtime_col
        # ].agg(lambda x: (x.iloc[-1] - x.iloc[-2]) / pd.Timedelta("1 minute"))
        labels_df = prefixes_df.groupby(self.prefixes_case_key, as_index=False, observed=True)[self.eventtime_col].agg(
            lambda x: (x.iloc[-1] - x.iloc[-2]) / pd.Timedelta("1 minute"))
        self.label_col = f"Transition time({activity_out}; {activity_in})"
        labels_df = labels_df.rename(columns={self.eventtime_col: self.label_col})
        return labels_df

    def _gen_label_next_activity(
            self, prefixes_df: pd.DataFrame, incoming_activities: Optional[Sequence[str]]
    ) -> pd.DataFrame:
        """Generate labels for the next activities

        :param prefixes_df: input DataFrame
        :param incoming_activities: the incoming activities for which to create
        labels. For all other incoming activities, one additional label is created.
        :return: label DataFrame
        """
        self.label_col = "label_next_Activity"
        labels = pd.DataFrame()
        labels[[self.prefixes_case_key, self.label_col]] = prefixes_df.groupby(
            self.prefixes_case_key, observed=True, as_index=False
        ).apply(lambda x: x[self.activity_col].iloc[-1])

        # Rename incoming activities that are not in incoming_activities

        if incoming_activities is not None:
            labels.loc[
                ~labels[self.label_col].isin(incoming_activities), self.label_col
            ] = "OTHER"

        return labels

    def _remove_last_row(self, prefixes_df) -> pd.DataFrame:
        """Remove the last row of all case. This is needed for the decision points
        use case.

        :param prefixes_df: Prefixes DataFrame
        :return: DataFrame with removed last rows
        """
        prefixes_df = prefixes_df[
            prefixes_df.groupby(self.prefixes_case_key, observed=True).cumcount(ascending=False) > 0
            ]
        return prefixes_df

    def _adjust_eventtime_transition_times(
            self, prefixes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Set the eventtime of the last activity in a prefix-case to the same value
        as the previous activity. This is done to not have the time of the last
        transition as part of the past-time feature.

        :param prefixes_df: the prefixes DataFrame
        :return: DataFrame with adjusted eventtimes.
        """

        def overwrite(group):
            group[self.eventtime_col].iloc[-1] = group[self.eventtime_col].iloc[-2]
            return group

        prefixes_df = prefixes_df.groupby(self.activity_case_key, observed=True).apply(overwrite)
        return prefixes_df

    def _gen_prefixes_transitions(
            self,
            df: pd.DataFrame,
            activity_out: Optional[List[str]] = None,
            activity_in: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate prefixes for transition times. This is needed because a
        transition can happen multiple times. The output DataFrame's prefix-cases
        will always end with the activity_in activity.

        :param df: activity DataFrame
        :param activity_out: outgoing activity
        :param activity_in: incoming activities. If None, all transitions from
        activity_out are taken

        return: prefix DataFrame
        """
        if activity_in is None:
            activity_in = df[self.activity_col].unique()
        if activity_out is None:
            activity_out = df[self.activity_col].unique()
        # Identify the rows of the transitions (row of incoming activity)
        mask_transitions = (
                ((df[self.activity_col].isin(activity_out)).shift(1))
                & (df[self.activity_col].isin(activity_in))
                & (df[self.activity_case_key].shift(1) == df[self.activity_case_key])
        )

        # To binary
        df_mask = pd.DataFrame(df[self.activity_case_key])
        df_mask["bin_transitions"] = mask_transitions
        # cumulate backwards
        df_mask = (
            df_mask.iloc[::-1]
                .groupby(self.activity_case_key, observed=True, as_index=False)["bin_transitions"]
                .cumsum()[::-1]
        )
        # Get the maximum number of these transitions (minus 1)
        max_transitions = int(df_mask["bin_transitions"].max())

        df_prefixes = pd.DataFrame(
            columns=df.columns.tolist() + [self.prefixes_case_key, "num_activities"]
        )

        def num_activities(group):
            g = group[self.activity_case_key].size
            group["num_activities"] = g
            return group

        self.static_numerical_cols.append("num_activities")
        for i in range(1, max_transitions + 1):
            tmp = df[df_mask["bin_transitions"] >= i]
            tmp = tmp.groupby(self.activity_case_key, as_index=False, observed=True).apply(
                num_activities
            )
            tmp[self.prefixes_case_key] = tmp[self.activity_case_key].apply(
                lambda x: f"{x}_{i}"
            )
            df_prefixes = pd.concat([df_prefixes, tmp], axis=0)
        return df_prefixes

    def _query_datamodel_transition(
            self, activity_out: Optional[List[str]], activity_in: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get activity and case table as DataFrames. Only the cases are queried that
        have a transition from activity_1 to activity_2

        :param activity_out: outgoing activity
        :param activity_in: incoming activites. If None, all activities are chosen
        as possible incoming activities.
        return: activity and case DataFrame
        """

        columns_activities = [item["name"] for item in self.activity_table.columns]
        columns_cases = [item["name"] for item in self.case_table.columns]

        # Queries for all columns
        query_activities = PQL()
        for col in columns_activities:
            query_activities += PQLColumn(
                '"' + self.activity_table.name + '"' + "." + '"' + col + '"', col
            )

        query_cases = PQL()
        for col in columns_cases:
            query_cases += PQLColumn(
                '"' + self.case_table.name + '"' + "." + '"' + col + '"', col
            )

        # Add filter for transition
        if activity_in is None:
            term_activity_in = 'ANY'
        else:
            term_activity_in = (
                    "(" + ", ".join("'" + act + "'" for act in activity_in) + ")"
            )

        if activity_out is None:
            term_activity_out = 'ANY'
        else:
            term_activity_out = (
                    "(" + ", ".join("'" + act + "'" for act in activity_out) + ")"
            )
        query_cases += PQLFilter(
            f"PROCESS EQUALS {term_activity_out} TO {term_activity_in}"
        )
        query_activities += PQLFilter(
            f"PROCESS EQUALS {term_activity_out} TO {term_activity_in}"
        )

        activity_df = self.dm.get_data_frame(query_activities)

        if self.case_table:
            case_df = self.dm.get_data_frame(query_cases)
        else:
            case_df = pd.DataFrame()
        return activity_df, case_df

    def _add_start_end_activities(self, activity_df: pd.DataFrame) -> pd.DataFrame:
        """Add a Start and End activity to each case. The eventtime of the start
        activity will be the same as the eventtime of the following activity minus
        1ns. The eventtime of the end activity will be the same as the eventtime of
        the previous activity plus 1ns.
        """

        def gen_start_end_row(group):
            df_new = pd.DataFrame(
                {
                    self.activity_case_key: [group.iloc[0][self.activity_case_key]] * 2,
                    self.activity_col: [self.activity_start, self.activity_end],
                    self.eventtime_col: [
                        group.iloc[0][self.eventtime_col] - pd.Timedelta(1),
                        group.iloc[-1][self.eventtime_col] + pd.Timedelta(1),
                    ],
                }
            )
            complete_df = pd.concat(
                [pd.DataFrame(df_new.iloc[0]).T, group, pd.DataFrame(df_new.iloc[1]).T]
            )
            return complete_df

        df_added = activity_df.groupby(self.activity_case_key, observed=True, as_index=False).apply(
            gen_start_end_row
        )
        # df_added = df_added.droplevel(0).reset_index(drop=True)
        return df_added

    def _query_datamodel(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get activity and case table as DataFrames. The whole tables are queried

        return: activity and case DataFrame
        """

        activity_df = self.activity_table.get_data_frame(chunksize=self.chunksize)
        if self.case_table:
            case_df = self.case_table.get_data_frame(chunksize=self.chunksize)
        else:
            case_df = pd.DataFrame()

        return activity_df, case_df

    def _query_datamodel_only_case(self) -> pd.DataFrame:
        """Get case table as DataFrames. The whole table is queried

        return: case DataFrame
        """
        case_df = self.case_table.get_data_frame(chunksize=self.chunksize)

        return case_df

    def _gen_labels_conforming(self, df: pd.DataFrame, case_key_col: str,
                               case_keys_conforming: np.ndarray) -> pd.DataFrame:
        """ Generate labels for conforming cases ('conforming' and 'not conforming')

        :param case_df: input DataFrame
        :param case_key_col: name of the case key column
        :param case_key_conforming: are with the case_keys of the conforming cases
        :return: the DataFrame with the case keys and labels
        """
        self.label_col = 'label_conforming'

        def gen_conforming(g):
            if g[case_key_col].values[0] in case_keys_conforming:
                g[self.label_col] = 1
                return g[[case_key_col, self.label_col]]
            else:
                g[self.label_col] = 0
                return g[[case_key_col, self.label_col]]

        labels = df.groupby(case_key_col, as_index=False, observed=True).apply(gen_conforming)

        return labels

    def _query_case_keys_conforming(self, analysis: Analysis, shared_url: str) -> np.ndarray:
        """ Query the case keys of the cases that are conforming to the selection in the shared url.

        :param analysis: the analysis from which to use the shared_url
        :param shared_url: the shared url from the analysis
        :return: array with the case key from the case table of the conforming cases
        """
        pql_shared = analysis.process_shared_selection_url(shared_url)
        query = PQL()
        query += PQLColumn('"' + self.case_table_name + '"' + "." + '"' + self.case_case_key + '"', self.case_case_key)
        query += pql_shared
        dm = analysis.datamodel
        df = self.dm.get_data_frame(query)
        case_keys = df[self.case_case_key].values
        return case_keys

    def _remove_activity_status(self, activity_df: pd.DataFrame):
        if 'lifecycle:transition' in activity_df.columns:

            activity_df = activity_df[activity_df['lifecycle:transition'] == 'COMPLETE']
            activity_df = activity_df.drop('lifecycle:transition', axis=1)
            return activity_df
        else:
            return activity_df

    def _set_latest_date(self):
        query = PQL()
        query += PQLColumn("MAX(" + '"' + self.activity_table_name + '"' + "." + '"' + self.eventtime_col + '"' + ")",
                           'latest_date')
        df = self.dm.get_data_frame(query)
        self.latest_date = df['latest_date'][0]

    def _processed_case_table_train(
            self
    ) -> pd.DataFrame:
        """Process case DataFrame for training data. This sets the
        member variables self.static_numerical_cols and
        self.static_categorical_cols.
        It processes the tables in the following order:
        1. Get DataFrame from the Database
        2. Convert object dtypes to categorical
        3. Remove datetime columns
        4. Rename case columns to "<casetable name>_<column_name>"
        5. Define static columns (the member variables)
        6. Remove few occurrences(defined by self.min_occurrences) of categorical
        columns except the activity column

        :return: processed case DataFrame
        """
        # Get DataFrame
        case_df = self._query_datamodel_only_case()

        # Convert object dtype to categorical
        case_df = self._conv_dtypes(case_df, ["object"], "category")

        # Remove datetime columns
        case_df = self._remove_datetime(case_df)

        # Rename case columns to "<casetable name>_<column_name>"
        self._rename_col_names(None, case_df)

        # Define static and dynamic columns
        self.static_numerical_cols = self._get_cols_by_type(
            case_df, dtypes=["number"], exclude=[self.case_case_key]
        )
        self.static_categorical_cols = self._get_cols_by_type(
            case_df, dtypes=["category"], exclude=[self.case_case_key]
        )

        # Remove few occurrences of categorical columns
        dynamic_cat_cols_without_activity = filter(
            lambda x: x != self.activity_col, self.dynamic_categorical_cols
        )

        case_df = self._remove_few_occurrences(
            case_df, self.static_categorical_cols, self.min_occurrences
        )

        # Save the categorical values
        self._save_categorical_values(None, case_df)

        return case_df

    def _processed_activity_case_tables_train(
            self, transition: Optional[Tuple[List[str], List[str]]] = None, inference: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process activity and case DataFrames for training data. This sets the
        member variables self.static_numerical_cols,
        self.static_categorical_cols, self.dynamic_numerical_cols,
        and self.dynamic_categorical_cols.
        It processes the tables in the following order:
        1. Get DataFrames from the Database
        2. Convert object dtypes to categorical
        3. Remove datetime columns except the eventtime column in activity_df
        4. Remove sorting column from activity table
        5. Rename activity columns to "<activitytable name>_<column_name>" and case
        columns to "<casetable name>_<column_name>"
        6. Define static and dynamic columns (the member variables)
        7. Remove few occurrences(defined by self.min_occurrences) of categorical
        columns except the activity column

        :param transition: Tuple with outgoing and incoming activity. Used if only data
        with such a transition shall be queried.
        :param inference: if True, the loaded data from the json config is used.
        :return: processed activity and case DataFrames
        """
        # Get DataFrames
        if transition:
            activity_df, case_df = self._query_datamodel_transition(
                transition[0], transition[1]
            )
        else:
            activity_df, case_df = self._query_datamodel()

        # Convert object dtype to categorical
        activity_df = self._conv_dtypes(activity_df, ["object"], "category")
        case_df = self._conv_dtypes(case_df, ["object"], "category")

        # Remove all transitions that are not completed
        activity_df = self._remove_activity_status(activity_df)

        # Save latest eventtime date
        self.latest_date = activity_df[self.eventtime_col].max()

        # Remove datetime columns except the eventtime column in activity_df
        activity_df = self._remove_datetime(activity_df, self.eventtime_col)
        case_df = self._remove_datetime(case_df)
        # Remove sorting column from activity table
        if self.sort_col is not None:
            activity_df.drop(self.sort_col, inplace=True, axis=1)

        # Rename activity columns to "<activitytable name>_<column_name>"
        # and case columns to "<casetable name>_<column_name>"
        self._rename_col_names(activity_df, case_df)

        # Define static and dynamic columns

        self.static_numerical_cols = self._get_cols_by_type(
            case_df, dtypes=["number"], exclude=[self.case_case_key]
        )
        self.dynamic_numerical_cols = self._get_cols_by_type(
            activity_df, dtypes=["number"], exclude=[self.activity_case_key]
        )
        self.static_categorical_cols = self._get_cols_by_type(
            case_df, dtypes=["category"], exclude=[self.case_case_key]
        )

        self.dynamic_categorical_cols = self._get_cols_by_type(
            activity_df, dtypes=["category"], exclude=[self.activity_case_key]
        )

        # Remove few occurrences of categorical columns except the activity column (if inference = False, else also for activity column)

        if inference:
            activity_df = self._remove_few_occurrences_inference(
                activity_df, self.dynamic_categorical_values
            )

            case_df = self._remove_few_occurrences_inference(
                case_df, self.static_categorical_values
            )
        else:
            dynamic_cat_cols_without_activity = filter(
                lambda x: x != self.activity_col, self.dynamic_categorical_cols
            )
            activity_df = self._remove_few_occurrences(
                activity_df, dynamic_cat_cols_without_activity, self.min_occurrences
            )

            case_df = self._remove_few_occurrences(
                case_df, self.static_categorical_cols, self.min_occurrences
            )

        # Save the categorical values if inference == False
        if not inference:
            self._save_categorical_values(activity_df, case_df)

        return activity_df, case_df

    def _load_config(self, filename: str):
        """ Load config from a json file
        :param filename: jilename of the json file without extension.
        :return:
        """
        with open(filename + '.json') as json_file:
            data = json.load(json_file)
            self.final_columns = data['final_columns']
            self.static_categorical_values = data['static_categorical_values']
            self.dynamic_categorical_values = data['dynamic_categorical_values']

    def _save_config(self, df: pd.DataFrame, filename: str):
        """ Save the preprocessing configuration to a json file.
        :param df: input DataFrame
        :param filename: filename of the json file without extension.
        :return:
        """
        data = {}
        data['final_columns'] = df.columns.tolist()
        data['static_categorical_values'] = self.static_categorical_values
        data['dynamic_categorical_values'] = self.dynamic_categorical_values
        with open(filename + '.json', 'w') as outfile:
            json.dump(data, outfile)

    def run_future_activity_training(self, activity_name: str) -> pd.DataFrame:
        """Runs the preprocessing for the use case of an activity happening in the
        future for training data.

        :param activity_name: the name of the activity to investigate
        :return: DataFrame with the preprocessed training data
        """
        activity_df, case_df = self._processed_activity_case_tables_train()
        # Merge activity and case DataFrames
        if self.case_table:
            joined_df = self._join_dfs(
                [activity_df, case_df], [self.activity_case_key, self.case_case_key]
            )
        else:
            joined_df = activity_df
        # Generate Prefixes
        joined_prefixes_df = self._gen_prefixes(
            joined_df,
            self.activity_case_key,
            self.prefixes_case_key,
            self.min_prefixes,
            self.max_prefixes,
        )
        # Generate labels
        labels_df = self._gen_label_future_activity(
            activity_df, joined_prefixes_df, activity_name
        )
        # Aggregate static and dynamic values
        aggregate_static_categorical_df = self._aggregate_static_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_categorical_cols,
        )

        aggregate_static_numerical_df = self._aggregate_static_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_numerical_cols,
        )
        aggregate_dynamic_categorical_df = self._aggregate_dynamic_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_categorical_cols,
            aggregations=self.aggregations_dyn_cat,
        )
        aggregate_dynamic_numerical_df = self._aggregate_dynamic_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_numerical_cols,
            aggregations=self.aggregations_dyn_num,
        )
        aggregate_time_past_df = self._compute_past_time(
            joined_prefixes_df, self.prefixes_case_key, self.eventtime_col
        )
        # Join the aggregated DataFrames
        if self.case_table:
            aggregated_df = self._join_dfs(
                [
                    aggregate_static_categorical_df,
                    aggregate_static_numerical_df,
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 6,
            )
        else:
            aggregated_df = self._join_dfs(
                [
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 4,
            )
        return aggregated_df

    def run_remaining_time(self, inference=False) -> pd.DataFrame:
        """Runs the preprocessing for the use case of predicting the remaining time for
        training data.

        :return: DataFrame with the preprocessed training data
        """
        # Load config data from json file if inference == True
        self.config_file_name = self.dm.name + "__remaining_time"
        if inference:
            self._load_config(self.config_file_name)

        activity_df, case_df = self._processed_activity_case_tables_train(inference=inference)

        # Merge activity and case DataFrames
        if self.case_table:
            joined_df = self._join_dfs(
                [activity_df, case_df], [self.activity_case_key, self.case_case_key]
            )
        else:
            joined_df = activity_df
        # Generate Prefixes
        joined_prefixes_df = self._gen_prefixes(
            joined_df,
            self.activity_case_key,
            self.prefixes_case_key,
            self.min_prefixes,
            self.max_prefixes,
        )

        # Generate labels
        labels_df = self._gen_label_remaining_execution_time(
            activity_df, joined_prefixes_df
        )
        # Aggregate static and dynamic values
        aggregate_static_categorical_df = self._aggregate_static_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_categorical_cols,
        )
        aggregate_static_numerical_df = self._aggregate_static_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_numerical_cols,
        )

        aggregate_dynamic_categorical_df = self._aggregate_dynamic_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_categorical_cols,
            aggregations=self.aggregations_dyn_cat,
        )

        aggregate_dynamic_numerical_df = self._aggregate_dynamic_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_numerical_cols,
            aggregations=self.aggregations_dyn_num,
        )
        aggregate_time_past_df = self._compute_past_time(
            joined_prefixes_df, self.prefixes_case_key, self.eventtime_col
        )
        # Join the aggregated DataFrames
        if self.case_table:
            aggregated_df = self._join_dfs(
                [
                    aggregate_static_categorical_df,
                    aggregate_static_numerical_df,
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 6,
            )
        else:
            aggregated_df = self._join_dfs(
                [
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 4,
            )

        # Save json file if inference == False
        # Add missing columns if inference = True
        # if inference:
        #    aggregated_df = self._add_missing_columns(aggregated_df, final_columns)

        if not inference:
            self._save_config(aggregated_df, self.config_file_name)

        return aggregated_df

    def run_transition_time(
            self, activity_out: Optional[Sequence[str]], activity_in: Optional[Sequence[str]]
    ):
        activity_df, case_df = self._processed_activity_case_tables_train(
            (activity_out, activity_in)
        )

        # Merge activity and case DataFrames
        if self.case_table:
            joined_df = self._join_dfs(
                [activity_df, case_df], [self.activity_case_key, self.case_case_key]
            )
        else:
            joined_df = activity_df

        # Generate Prefixes
        joined_prefixes_df = self._gen_prefixes_transitions(
            joined_df,
            activity_out,
            activity_in,
        )

        # Generate labels
        labels_df = self._gen_label_transition_times(
            joined_prefixes_df, activity_out, activity_in
        )

        # Adjust eventtime
        joined_prefixes_df = self._adjust_eventtime_transition_times(joined_prefixes_df)

        # Aggregate static and dynamic values
        aggregate_static_categorical_df = self._aggregate_static_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_categorical_cols,
        )
        aggregate_static_numerical_df = self._aggregate_static_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_numerical_cols,
        )
        aggregate_dynamic_categorical_df = self._aggregate_dynamic_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_categorical_cols,
            aggregations=self.aggregations_dyn_cat,
        )
        aggregate_dynamic_numerical_df = self._aggregate_dynamic_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_numerical_cols,
            aggregations=self.aggregations_dyn_num,
        )
        aggregate_time_past_df = self._compute_past_time(
            joined_prefixes_df, self.prefixes_case_key, self.eventtime_col
        )
        # Join the aggregated DataFrames
        if self.case_table:
            aggregated_df = self._join_dfs(
                [
                    aggregate_static_categorical_df,
                    aggregate_static_numerical_df,
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 6,
            )
        else:
            aggregated_df = self._join_dfs(
                [
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 4,
            )
        return aggregated_df

    def run_decision_point(
            self, activity_out: str, activity_in: Optional[Sequence[str]]
    ):
        activity_out = [activity_out]

        activity_df, case_df = self._processed_activity_case_tables_train(
            (activity_out, None)
        )

        activity_df = self._add_start_end_activities(activity_df)
        # Merge activity and case DataFrames
        if self.case_table:
            joined_df = self._join_dfs(
                [activity_df, case_df], [self.activity_case_key, self.case_case_key]
            )
        else:
            joined_df = activity_df

        # Generate Prefixes
        joined_prefixes_df = self._gen_prefixes_transitions(
            joined_df,
            activity_out,
            None,
        )

        # Generate labels
        labels_df = self._gen_label_next_activity(joined_prefixes_df, activity_in)

        # Remove last activities from the joined DataFrame such that no future
        # information is used.
        joined_prefixes_df = self._remove_last_row(joined_prefixes_df)

        # Aggregate static and dynamic values
        aggregate_static_categorical_df = self._aggregate_static_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_categorical_cols,
        )
        aggregate_static_numerical_df = self._aggregate_static_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.static_numerical_cols,
        )
        aggregate_dynamic_categorical_df = self._aggregate_dynamic_categorical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_categorical_cols,
            aggregations=self.aggregations_dyn_cat,
        )
        aggregate_dynamic_numerical_df = self._aggregate_dynamic_numerical(
            joined_prefixes_df,
            case_key=self.prefixes_case_key,
            columns=self.dynamic_numerical_cols,
            aggregations=self.aggregations_dyn_num,
        )
        aggregate_time_past_df = self._compute_past_time(
            joined_prefixes_df, self.prefixes_case_key, self.eventtime_col
        )
        # Join the aggregated DataFrames
        if self.case_table:
            aggregated_df = self._join_dfs(
                [
                    aggregate_static_categorical_df,
                    aggregate_static_numerical_df,
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 6,
            )
        else:
            aggregated_df = self._join_dfs(
                [
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    aggregate_time_past_df,
                    labels_df,
                ],
                [self.prefixes_case_key] * 4,
            )
        return aggregated_df

    def run_total_time_training(self) -> pd.DataFrame:
        """Runs the preprocessing for the use case of explaining the total trace time for
        training data.

        :return: DataFrame with the preprocessed training data
        """
        activity_df, case_df = self._processed_activity_case_tables_train()
        # Merge activity and case DataFrames
        if self.case_table:
            joined_df = self._join_dfs(
                [activity_df, case_df], [self.activity_case_key, self.case_case_key]
            )
        else:
            joined_df = activity_df

        # Generate labels
        labels_df = self._gen_label_total_time(
            joined_df, self.activity_case_key
        )

        # Aggregate static and dynamic values
        aggregate_static_categorical_df = self._aggregate_static_categorical(
            joined_df,
            case_key=self.activity_case_key,
            columns=self.static_categorical_cols,
        )
        aggregate_static_numerical_df = self._aggregate_static_numerical(
            joined_df,
            case_key=self.activity_case_key,
            columns=self.static_numerical_cols,
        )
        aggregate_dynamic_categorical_df = self._aggregate_dynamic_categorical(
            joined_df,
            case_key=self.activity_case_key,
            columns=self.dynamic_categorical_cols,
            aggregations=self.aggregations_dyn_cat,
        )
        aggregate_dynamic_numerical_df = self._aggregate_dynamic_numerical(
            joined_df,
            case_key=self.activity_case_key,
            columns=self.dynamic_numerical_cols,
            aggregations=self.aggregations_dyn_num,
        )

        # Join the aggregated DataFrames
        if self.case_table:
            aggregated_df = self._join_dfs(
                [
                    aggregate_static_categorical_df,
                    aggregate_static_numerical_df,
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    labels_df,
                ],
                [self.activity_case_key] * 5,
            )
        else:
            aggregated_df = self._join_dfs(
                [
                    aggregate_dynamic_categorical_df,
                    aggregate_dynamic_numerical_df,
                    labels_df,
                ],
                [self.activity_case_key] * 3,
            )
        return aggregated_df

    def run_deviations(self, analysis, shared_url) -> pd.DataFrame:
        """Runs the preprocessing for the use case of explaining the total trace time for
        training data.

        :return: DataFrame with the preprocessed training data
        """
        case_df = self._processed_case_table_train()

        # Get case keys of conforming cases
        case_keys_conforming = self._query_case_keys_conforming(analysis, shared_url)

        # Generate labels
        labels_df = self._gen_labels_conforming(
            case_df, self.case_case_key, case_keys_conforming
        )

        # Aggregate static values
        aggregate_static_categorical_df = self._aggregate_static_categorical(
            case_df,
            case_key=self.case_case_key,
            columns=self.static_categorical_cols,
        )

        aggregate_static_numerical_df = self._aggregate_static_numerical(
            case_df,
            case_key=self.case_case_key,
            columns=self.static_numerical_cols,
        )

        # Join the aggregated DataFrames
        aggregated_df = self._join_dfs(
            [
                aggregate_static_categorical_df,
                aggregate_static_numerical_df,
                labels_df,
            ],
            [self.case_case_key] * 3,
        )

        return aggregated_df
