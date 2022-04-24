from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import wittgenstein as lw
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix

from one_click_analysis.errors import MaximumValueReachedError
from one_click_analysis.errors import MinimumValueReachedError


class DecisionRuleMiner:
    """Wrapper class for the wittgenstein.RIPPER algorithm with additional
    functionality.
    Ripper implementation source code: https://github.com/imoscovitz/wittgenstein
    """

    configs = [
        {"max_rule_conds": 1, "max_rules": 1},
        {"max_rule_conds": 1, "max_rules": 2},
        {"max_rule_conds": 2, "max_rules": 2},
        {"max_rule_conds": 2, "max_rules": 3},
        {"max_rule_conds": 2, "max_rules": 6},
        {"max_rule_conds": 3, "max_rules": 6},
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        class_label: str,
        attrbute_labels: List[str],
        pos_class: Optional[str] = None,
        threshold: Optional[float] = None,
        config_index: int = 2,
        n_discretize_bins: int = 3,
        k: int = 3,
        random_state: int = 42,
    ):
        """

        :param df: DataFrame with independent variables and dependent variables
        :param class_label: column name of the dependent varialbe in df
        :param attrbute_labels: column names of the independent variables
        :param pos_class: value of the positive class of the dependent variable
        :param threshold: if the dependent variable is numeric, the values
        above or equal to the threshold are labeled as the positive class
        :param config_index: the index of the configs to start with
        :param n_discretize_bins: number of bins to divide the numerical independent
        variables
        :param k: number of iterations to run the ripper algorithm
        :param random_state: random state seed to get reproducible results
        """
        self.df = df
        self.preprocessed_label_col = "Decisionrulelabel"
        self.class_label = class_label
        self.attribute_labels = attrbute_labels
        # probably shouldn't change this to a higher value since it will become too
        self.max_rule_conds = None
        # difficult to read
        self.n_discretize_bins = (
            n_discretize_bins
            # this should be fixed. Maybe can also use 3
        )
        # the value of the positive class. Leave at None if class is numerical
        self.pos_class = pos_class
        self.k = k  # number of iterations
        self.random_state = random_state
        self.max_rules = (
            None
            # this variable can be changed to make rules simpler or
            # more elaborate
        )
        # threshold to define positive class for numerical class. Positive class is
        # >= threshold
        self.threshold = threshold
        # threshold
        self.train_df = self._gen_train_df()
        self.rules = None
        self.attr_dict = (
            self._create_attribute_dict()
        )  # dict that maps from the changed names
        # to the original names
        self.clf = None
        self.structured_rules = None
        self.metrics = {}
        self.config_index = config_index
        self.apply_config()

    def apply_config(self):
        """Apply the config at index self.config_index to the corresponding member
        variables.

        :return:
        """
        self.max_rule_conds = self.configs[self.config_index]["max_rule_conds"]
        self.max_rules = self.configs[self.config_index]["max_rules"]
        self.config_index = self.config_index

    def _gen_label_df(self) -> pd.DataFrame:
        """Generate the DataFrame with a column with the positive(1) and negative(0)
        classes of the dependent variable from the label column in self.df.

        :return: DataFrame with column of the positive and negative classes
        """
        label_df = pd.DataFrame(index=self.df.index)
        # create new label column if original is numerical
        if is_numeric_dtype(self.df[self.class_label]) and self.threshold is not None:
            label_df[self.preprocessed_label_col] = np.where(
                self.df[self.class_label] >= self.threshold, 1, 0
            )
            self.pos_class = 1

        else:
            label_df[self.preprocessed_label_col] = self.df[self.class_label]
        return label_df

    def _gen_train_df(self):
        """create DataFrame used for training"""
        label_df = self._gen_label_df()
        df_train = self.df[self.attribute_labels]
        df_train[self.preprocessed_label_col] = label_df[self.preprocessed_label_col]
        return df_train

    def run_pipeline(self):
        """Run the pipeline that fits the ripper model and computes metrics."""
        self._fit()
        self.structured_rules = self.create_structured_rules()
        pred = self.make_predictions()
        (
            true_n,
            false_p,
            false_n,
            true_p,
            recall_p,
            recall_n,
            precision_p,
            precision_n,
        ) = self._get_confusion_matrix(pred)
        avg_True, avg_False = self._get_avg_label_value(pred)
        metric_dict = {
            "true_n": true_n,
            "false_p": false_p,
            "false_n": false_n,
            "true_p": true_p,
            "recall_p": recall_p,
            "recall_n": recall_n,
            "precision_p": precision_p,
            "precision_n": precision_n,
            "avg_True": avg_True,
            "avg_False": avg_False,
        }
        self.metrics = metric_dict

    def _fit(self):
        """Create a RIPPER model object and fit it.

        :return:
        """
        self.clf = lw.RIPPER(
            max_rule_conds=self.max_rule_conds,
            n_discretize_bins=self.n_discretize_bins,
            k=self.k,
            random_state=self.random_state,
            max_rules=self.max_rules,
        )

        self.clf.fit(
            self.train_df,
            class_feat=self.preprocessed_label_col,
            pos_class=self.pos_class,
        )
        self.rules = self.clf.ruleset_

    def make_predictions(self) -> np.ndarray:
        """Make predictions for all data in self.df with the fitted model.

        :return:array with the predicted classes
        """
        pred = self.clf.predict(self.train_df[self.attribute_labels])
        return np.array(pred)

    def _get_confusion_matrix(
        self, pred
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """Get model metrics on the predictions.
        The metrics are:
        true-negative samples,
        false-positive samples,
        false-negative samples,
        true-positive samples,
        recall for positive class,
        recall for negative class,
        precision for positive class,
        precision for negative class

        :param pred: array with predictions
        :return: Tuple with the metrics
        """
        true_labels = self.train_df[self.preprocessed_label_col].values
        # remove nan rows
        idxs_nan = np.argwhere(np.isnan(true_labels))
        mask = np.full(len(true_labels), True)
        mask[idxs_nan] = False
        true_labels = true_labels[mask]
        pred = pred[mask]
        true_n, false_p, false_n, true_p = confusion_matrix(true_labels, pred).ravel()
        recall_p = true_p / (true_p + false_n)
        recall_n = true_n / (true_n + false_p)
        precision_p = true_p / (true_p + false_p)
        precision_n = true_n / (true_n + false_n)
        return (
            true_n,
            false_p,
            false_n,
            true_p,
            recall_p,
            recall_n,
            precision_p,
            precision_n,
        )

    def _get_avg_label_value(self, pred: np.ndarray):
        """Get the average values of the independent variable for the cases that are
        predicted to be in the positive and the negative class

        :param pred: array with predictions
        :return: average values of independent variable for the cases that are
        predicted to be in the positive and the negative class
        """
        avg_True = self.df[pred][self.class_label].mean()
        avg_False = self.df[~pred][self.class_label].mean()
        return avg_True, avg_False

    def create_structured_rules(self) -> List[List[Dict[str, str]]]:
        """Create a list with rules in an easier to process format from the raw
        decision rules of the RIPPER model.

        :return: list with the structured rues
        """
        ruleset = self.clf.ruleset_
        structured_rules = []
        for rule in ruleset:
            conds = []
            for cond in rule.conds:
                feature = self.attr_dict[cond.feature]
                val = cond.val
                unequal_sign = ""
                if isinstance(val, str) and "<" in val:
                    unequal_sign = "<"
                    val = val.replace("<", "")
                elif isinstance(val, str) and ">" in val:
                    unequal_sign = ">"
                    val = val.replace(">", "")
                elif isinstance(val, str):
                    unequal_sign = "between"

                conds.append(
                    (
                        {
                            "attribute": feature,
                            "value": str(val),
                            "unequal_sign": unequal_sign,
                        }
                    )
                )
            structured_rules.append(conds)
        return structured_rules

    def _create_attribute_dict(self) -> Dict[str, str]:
        """Inside the decision miner algorythm, the names of the original attributes
        are changed a bit. So here a dict is created that maps from the changed names to
        the original names

        :return: dictionary mapping the changed attribute names from the RIPPER
        algorithm to the original attribute names
        """
        attr_dict = {}
        for attr in self.attribute_labels:
            changed_str = attr.replace("'", "")
            attr_dict[changed_str] = attr
        return attr_dict

    def simplify_rule_config(self):
        """Reduce the config by 1

        :raise MinimumValueReachedError: if decreasing the config_index would
        decrease it to a number below 0.
        :return:
        """
        if self.config_index > 0:
            self.config_index -= 1
            self.apply_config()
        else:
            raise MinimumValueReachedError()

    def elaborate_rule_config(self):
        """Reduce the config by 1

        :raise MaximumValueReachedError: if increasing the config_index would exceed
        the maximum index in self.configs
        :return:
        """
        if self.config_index < (len(self.configs) - 1):
            self.config_index += 1
            self.apply_config()
        else:
            raise MaximumValueReachedError()
