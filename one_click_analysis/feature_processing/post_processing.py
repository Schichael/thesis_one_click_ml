from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.common import is_string_dtype

from one_click_analysis import utils
from one_click_analysis.errors import WrongFeatureTypeError
from one_click_analysis.feature_processing.attributes.attribute import Attribute
from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDataType,
)
from one_click_analysis.feature_processing.attributes.feature import Feature


class PostProcessor:
    """Post processes a dataframe and generates a list of Feature objects."""

    def __init__(
        self,
        df_x: pd.DataFrame,
        df_target: pd.DataFrame,
        attributes: List[Attribute],
        target_attributes: Union[Attribute, List[Attribute]],
        valid_target_values: Optional[List[str]] = None,
        invalid_target_replacement: Optional[str] = None,
        min_counts_perc: float = 0.0,
        max_counts_perc: float = 1.0,
    ):
        """

        :param df_x:
        :param df_target:
        :param attributes:
        :param target_attributes: target attributes.
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
        self.attributes = attributes
        self.target_attributes = utils.make_list(target_attributes)
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
        df_target. This does not include the removal of NaN values.

        :return: processed df_x, df_target, feature lists for target and normal features
        """
        # process target df
        target_features = self.process_target_attributes()
        features = self.process_feature_attributes(self.attributes)
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
            if not attr.is_feature:
                continue
            self.validate_attr_datatype(attr)
            # OHE if categorical and not yet encoded
            if attr.data_type == AttributeDataType.CATEGORICAL and is_string_dtype(
                self.df_x[attr.attribute_name]
            ):
                # Remove too few and too many occurences (set them to np.nan)
                self.df_x.loc[
                    (
                        self.df_x[attr.attribute_name]
                        .value_counts(dropna=False)[self.df_x[attr.attribute_name]]
                        .values
                        <= self.min_counts
                    )
                    | (
                        self.df_x[attr.attribute_name]
                        .value_counts(dropna=False)[self.df_x[attr.attribute_name]]
                        .values
                        >= self.max_counts
                    ),
                    attr.attribute_name,
                ] = np.nan

                # One hot encode
                prefix = attr.attribute_name
                prefix_sep = " = "
                df_attr_cols = pd.get_dummies(
                    self.df_x[attr.attribute_name],
                    prefix=prefix,
                    prefix_sep=prefix_sep,
                )

                # update target df
                self._update_df(df=self.df_x, new_cols_df=df_attr_cols, attr=attr)

                features_attr = self._create_features(
                    df=df_attr_cols, attr=attr, prefix=prefix + prefix_sep
                )
                feature_list = feature_list + features_attr
            else:
                features_attr = self._create_features(
                    df=pd.DataFrame(self.df_x[attr.attribute_name]), attr=attr
                )
                feature_list = feature_list + features_attr
        return feature_list

    def process_target_attributes(self):
        """One-hot-encode the target feature"""
        # TODO: Also validate min and max values from already ohe'd columns
        target_features = []
        for target_attr in self.target_attributes:
            target_col_name = target_attr.attribute_name
            # Check whether the target attributes are numerical or categorical
            self.validate_attr_datatype(target_attr)
            attr_datatype = target_attr.data_type
            if attr_datatype == AttributeDataType.CATEGORICAL and not is_numeric_dtype(
                self.df_target[target_col_name]
            ):
                # Handle invalid target value names
                if self.valid_target_values:
                    if self.invalid_target_replacement is None:
                        # Set invalid values to NAN
                        self.df_target.loc[
                            self.df_target[target_col_name]
                            not in self.valid_target_values
                        ] = np.nan
                    else:
                        self.df_target.loc[
                            ~self.df_target[target_col_name].isin(
                                self.valid_target_values
                            )
                        ] = self.invalid_target_replacement

                # One hot encoding
                prefix = target_col_name
                prefix_sep = " = "
                df_target_cols = pd.get_dummies(
                    self.df_target[target_col_name],
                    prefix=prefix,
                    prefix_sep=prefix_sep,
                )

                target_features = self._create_features(
                    df=df_target_cols,
                    attr=target_attr,
                    prefix=prefix + prefix_sep,
                )

                # update target df
                self._update_df(
                    df=self.df_target,
                    new_cols_df=df_target_cols,
                    attr=target_attr,
                )
            else:
                target_features += self._create_features(
                    df=pd.DataFrame(self.df_target[target_col_name]), attr=target_attr
                )
            # Remove old column in df
            # Set new columns
            # self.df_target[df_target_cols.columns] = df_target_cols
        return target_features

    def _update_df(self, df: pd.DataFrame, new_cols_df: pd.DataFrame, attr: Attribute):
        """Remove old attribute column and add new columns from new_cols_df to df"""
        # Remove old column in df
        df.drop(attr.attribute_name, axis=1, inplace=True)
        # Set new columns
        df[new_cols_df.columns] = new_cols_df

    def _create_features(
        self, df: pd.DataFrame, attr: Attribute, prefix: str = None
    ) -> List:
        """

        :param df:
        :param attr:
        :param prefix: prefix before the actual value name if feature was created
        from ohe.
        :return:
        """
        features = []

        for col in df.columns:
            if prefix:
                attribute_value = col[len(prefix) :]
            elif attr.value is not None:
                attribute_value = attr.value
            else:
                attribute_value = None
            feature = Feature(
                df_column_name=col,
                datatype=attr.data_type,
                attribute=attr,
                attribute_value=attribute_value,
                unit=attr.unit,
            )
            features.append(feature)
        return features

    def compute_min_max_attribute_counts_PQL(
        self, min_counts_perc: float, max_counts_perc: float
    ):
        num_rows = len(self.df_target.index)
        min_counts = round(min_counts_perc * num_rows)
        max_counts = round(max_counts_perc * num_rows)
        return min_counts, max_counts


def remove_nan(
    df_x: pd.DataFrame,
    df_target: pd.DataFrame,
    features: List[Feature],
    target_features: List[Feature],
    th_remove_col: float = 0.3,
):
    """Remove nan values.
    For categorical features: set to 0 if NaN. If too many values are nan (more than
    th_remove_col times the number of rows),
    remove the column
    For numeric features: if column has more NaN values that th_remove_col of the
    rows, the whole column is removed. This also removes the feature from the feature
    list.
    For target feature: if target feature(s) value is nan, remove whole row. If not
    too many NaN value, replace nan values with the median value
    """
    # Get column names with nan values
    feature_dict = {f.df_column_name: f for f in features}
    target_feature_dict = {f.df_column_name: f for f in target_features}
    # remove rows with nan values in df_target
    df_target_only_features = df_target[list(target_feature_dict.keys())]
    indices_nan = np.where(pd.isnull(df_target_only_features).any(1))[0]
    df_x = df_x[~df_x.index.isin(indices_nan)]
    df_target = df_target[~df_target.index.isin(indices_nan)]
    cols_with_nan = df_x.columns[df_x.isna().any()].tolist()

    for col in cols_with_nan:
        if feature_dict.get(col) is None:
            continue
        if df_x[col].isna().sum() > len(df_x.index) * th_remove_col:
            df_x.drop(col, axis=1, inplace=True)
            features.remove(feature_dict[col])
            continue

        if feature_dict[col].datatype == AttributeDataType.CATEGORICAL:
            df_x[col] = df_x[col].fillna(value=0)

        if feature_dict[col].datatype == AttributeDataType.NUMERICAL:
            df_x[col] = df_x[col].fillna(value=df_x[col].median())
    return df_x, df_target
