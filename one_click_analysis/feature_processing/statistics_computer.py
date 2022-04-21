from one_click_analysis.feature_processing.attributes_new.attribute import (
    AttributeDataType,
)


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
            label_series = self.df_target[target_feature.column_name]
            corrs = self.df_x.corrwith(label_series)
            for feature in self.features:
                if "correlations" not in feature.metrics:
                    feature.metrics["correlations"] = {}
                for target_feature in self.target_features:
                    correlation = corrs[feature.column_name]
                    feature.metrics["correlations"][
                        target_feature.column_name
                    ] = correlation

    def case_count_with_feature(self):
        # Do this for both normal and target features
        feature_lists = [self.features, self.target_features]
        for feature_list in feature_lists:
            for feature in feature_list:
                if feature.datatype == AttributeDataType.CATEGORICAL:
                    case_count = len(
                        self.df_x[self.df_x[feature.column_name] == 1].index
                    )
                    feature.metrics["case_count"] = case_count

    def influence_on_target(self):
        for feature in self.features:
            if feature.datatype == AttributeDataType.CATEGORICAL:
                target_influences = {}
                for target_feature in self.target_features:
                    label_val_0 = self.df_target[self.df_x[feature.column_name] == 0][
                        target_feature.column_name
                    ].mean()
                    label_val_1 = self.df_target[self.df_x[feature.column_name] == 1][
                        target_feature.column_name
                    ].mean()
                    target_influences[target_feature.column_name] = (
                        label_val_1 - label_val_0
                    )
                feature.metrics["target_influence"] = target_influences
