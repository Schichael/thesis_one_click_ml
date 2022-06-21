import numpy as np

from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDataType,
)
from one_click_analysis.statistics.outlier_removal import remove_outliers_IQR


class StatisticsComputer:
    """Class to compute statistics on features."""

    def __init__(self, features, target_features, df_x, df_target):
        self.features = features
        self.target_features = target_features
        self.df_x = df_x
        self.df_target = df_target

    def compute_all_statistics(self):
        """computes all statistics inplace"""
        self.compute_correlations()
        self.case_count_with_feature()
        self.influence_on_target()

    def compute_correlations(self):
        """Compute correlations of the features with the target feature."""
        for target_feature in self.target_features:
            y_values = self.df_target[target_feature.df_column_name].values

            for feature in self.features:
                x_values = self.df_x[feature.df_column_name].values
                x_values_cleaned, y_values_cleaned = remove_outliers_IQR(
                    x_values, y_values, apply_on_binary=False
                )

                corr = np.corrcoef(x_values_cleaned, y_values_cleaned)[1, 0]
                if "correlations" not in feature.metrics:
                    feature.metrics["correlations"] = {}
                feature.metrics["correlations"][target_feature.df_column_name] = corr

    def case_count_with_feature(self):
        # Do this for both normal and target features
        for feature in self.features:
            if feature.datatype == AttributeDataType.CATEGORICAL:
                indices_with_feature = np.where(self.df_x[feature.df_column_name] == 1)[
                    0
                ]

                case_count = len(
                    set(
                        self.df_x.iloc[
                            indices_with_feature.tolist()
                        ].index.get_level_values(0)
                    )
                )
                feature.metrics["case_count"] = case_count

        for tf in self.target_features:
            if tf.datatype == AttributeDataType.CATEGORICAL:
                indices_with_feature = np.where(self.df_target[tf.df_column_name] == 1)[
                    0
                ]
                case_count = len(
                    set(
                        self.df_target.iloc[
                            indices_with_feature.tolist()
                        ].index.get_level_values(0)
                    )
                )
                tf.metrics["case_count"] = case_count

    def influence_on_target(self):
        for feature in self.features:
            if feature.datatype == AttributeDataType.CATEGORICAL:
                target_influences = {}
                for target_feature in self.target_features:
                    label_val_0 = self.df_target[
                        self.df_x[feature.df_column_name] == 0
                    ][target_feature.df_column_name].mean()
                    label_val_1 = self.df_target[
                        self.df_x[feature.df_column_name] == 1
                    ][target_feature.df_column_name].mean()
                    target_influences[target_feature.df_column_name] = (
                        label_val_1 - label_val_0
                    )
                feature.metrics["target_influence"] = target_influences
