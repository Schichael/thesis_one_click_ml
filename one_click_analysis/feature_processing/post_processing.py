from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.common import is_string_dtype

from one_click_analysis.errors import WrongFeatureTypeError
from one_click_analysis.feature_processing.attributes_new.attribute import (
    Attribute,
    AttributeDataType,
)
from one_click_analysis.feature_processing.attributes_new.dynamic_attributes import (
    DynamicAttribute,
)
from one_click_analysis.feature_processing.attributes_new.feature import Feature
from one_click_analysis.feature_processing.attributes_new.static_attributes import (
    StaticAttribute,
)


class PostProcessor:
    """Post processes a dataframe and generates a list of Feature objects."""

    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        static_attributes: List[StaticAttribute],
        dynamic_attributes: List[DynamicAttribute],
        target_attribute: Attribute,
        valid_target_values: Optional[List[str]] = None,
        invalid_target_replacement: Optional[str] = None,
        min_counts_perc: float = 0.0,
        max_counts_perc: float = 1.0,
    ):
        """

        :param df_x:
        :param df_target:
        :param static_attributes:
        :param dynamic_attributes:
        :param target_attribute: target attribute.
        :param valid_target_values: if the target attribute is categorical and not
        encoded with 0 and 1 yet, the values not in valid_target_values are encoded
        together to column TARGET_OTHER. If None, all values are used if
        remove_invalid_target_values is False

        :param invalid_target_replacement: string that replaces the invalid target
        values. If None, invalid target values are not replaced
        values, if they shall be replaced. If a value already
        :param min_counts_perc:
        :param max_counts_perc:
        """
        self.df_x = df_x
        self.df_target = df_target
        self.static_attributes = static_attributes
        self.dynamic_attributes = dynamic_attributes
        self.target_attribute = target_attribute
        self.valid_target_values = valid_target_values
        self.invalid_target_replacement = invalid_target_replacement
        self.min_counts, self.max_counts = self.compute_min_max_attribute_counts_PQL(
            min_counts_perc, max_counts_perc
        )

    def process(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Feature], List[Feature]]:
        """Processing pipeline.
        Creates features and target features. Processes the DataFrames df_x and
        df_target.

        :return: processed df_x, df_target, feature lists for target and normal features
        """
        # process target df
        target_features = self.process_target_attribute()
        static_features = self.process_feature_attributes(self.static_attributes)
        dynamic_features = self.process_feature_attributes(self.dynamic_attributes)
        features = static_features + dynamic_features
        return self.df_x, self.df_target, target_features, features

    def validate_attr_datatype(self, attr: Attribute):
        attr_datatype = attr.data_type
        if attr_datatype not in [
            AttributeDataType.NUMERICAL,
            AttributeDataType.CATEGORICAL,
        ]:
            raise WrongFeatureTypeError(attr.attribute_name, attr_datatype)

    def process_feature_attributes(self, attributes: List[Attribute]):
        feature_list = []
        for attr in attributes:
            self.validate_attr_datatype(attr)
            # OHE if categorical and not yet encoded
            if attr.data_type == AttributeDataType.CATEGORICAL and is_string_dtype(
                self.df_x[attr.attribute_name]
            ):
                # Remove too few occurences (set them to np.nan)
                self.df_x.loc[
                    (
                        self.df_x[attr.attribute_name]
                        .value_counts()[self.df_x[attr.attribute_name]]
                        .values
                        <= self.min_counts
                    )
                    & (
                        self.df_x[attr.attribute_name]
                        .value_counts()[self.df_x[attr.attribute_name]]
                        .values
                        >= self.max_counts
                    ),
                    attr.attribute_name,
                ] = np.nan

                # One hot encode
                df_attr_cols = pd.get_dummies(
                    self.df_x[attr.attribute_name],
                    prefix=attr.attribute_name,
                    prefix_sep=" = ",
                )
                features_attr = self._create_features(df=df_attr_cols, attr=attr)
                feature_list = feature_list + features_attr
            else:
                features_attr = self._create_features(
                    df=self.df_x[attr.attribute_name], attr=attr
                )
                feature_list = feature_list + features_attr
        return feature_list

    def process_target_attribute(self):
        """One-hot-encode the target feature"""
        target_col_name = self.target_attribute.attribute_name
        # Check whether the target attributes are numerical or categorical
        self.validate_attr_datatype(self.target_attribute)
        attr_datatype = self.target_attribute.data_type
        if attr_datatype == AttributeDataType.NUMERICAL:
            return
        # Check if the target is already encoded to be numerical
        if is_numeric_dtype(self.df_target[target_col_name]):
            return

        # Handle invalid target value names
        if self.valid_target_values:
            if self.invalid_target_replacement is None:
                # Set invalid values to NAN
                self.df_target.loc[
                    self.df_target[target_col_name] not in self.valid_target_values
                ] = np.nan
            else:
                self.df_target.loc[
                    self.df_target[target_col_name] not in self.valid_target_values
                ] = self.invalid_target_replacement

        # One hot encoding
        df_target_cols = pd.get_dummies(
            self.df_target[target_col_name], prefix=target_col_name, prefix_sep=" = "
        )

        target_features = self._create_features(
            df=df_target_cols, attr=self.target_attribute
        )

        # update target df
        self._update_df(
            df=self.df_target, new_cols_df=df_target_cols, attr=self.target_attribute
        )
        # Remove old column in df
        # self.df_target.drop(self.target_attribute.attribute_name, axis=1, inplace=True)
        # Set new columns
        # self.df_target[df_target_cols.columns] = df_target_cols
        return target_features

    def _update_df(self, df: pd.DataFrame, new_cols_df: pd.DataFrame, attr: Attribute):
        """Remove old attribute column and add new columns from new_cols_df to df"""
        # Remove old column in df
        df.drop(attr.attribute_name, axis=1, inplace=True)
        # Set new columns
        df[new_cols_df.columns] = new_cols_df

    def _create_features(self, df: pd.DataFrame, attr: Attribute):
        features = []
        for col in df.columns:
            feature = Feature(column_name=col, datatype=attr.data_type, attribute=attr)
            features.append(feature)
        return features

    def compute_min_max_attribute_counts_PQL(
        self, min_counts_perc: float, max_counts_perc: float
    ):
        num_rows = len(self.df_target.index)
        min_counts = round(min_counts_perc * num_rows)
        max_counts = round(max_counts_perc * num_rows)
        return min_counts, max_counts
