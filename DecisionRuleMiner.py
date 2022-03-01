import wittgenstein as lw
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pandas.api.types import is_numeric_dtype
from errors import MinimumValueReachedError, MaximumValueReachedError

class DecisionRuleMiner:
    configs = [{'max_rule_conds': 1, 'max_rules': 1}, {'max_rule_conds': 1, 'max_rules': 2},
        {'max_rule_conds': 2, 'max_rules': 2}, {'max_rule_conds': 2, 'max_rules': 3},
        {'max_rule_conds': 2, 'max_rules': 6}, {'max_rule_conds': 3, 'max_rules': 6}]

    def __init__(self, df, class_label, attrbute_labels, pos_class: str = None, threshold: float = None,
                 config_index:int = 2, n_discretize_bins: int = 3, k:int = 3, random_state:int = 42):
        self.df = df
        self.attrbute_labels = attrbute_labels
        self.preprocessed_label_col = "Decisionrulelabel"
        self.class_label = class_label
        self.attribute_labels = attrbute_labels
        self.max_rule_conds = None  # probably shouldn't change this to a higher value since it will become too
        # difficult to read
        self.n_discretize_bins = n_discretize_bins  # this should be fixed. Maybe can also use 3
        self.pos_class = pos_class  # the value of the positive class. Leave at None if class is numerical
        self.k = k  # number of iterations
        self.random_state = random_state
        self.max_rules = None  # this variable can be changed to make rules simpler or more elaborate

        self.threshold = threshold  # threshold to define positive class for numerical class. Positive class is >=
        # threshold
        self.label_df = self.get_label_df()
        self.rules = None
        self.attr_dict = self._create_attribute_dict()  # dict that maps from the changed names to the original names
        self.clf = None
        self.structured_rules = None
        self.metrics = {}
        self.config_index = config_index
        self.apply_config()

    def apply_config(self):
        self.max_rule_conds = self.configs[self.config_index]['max_rule_conds']
        self.max_rules = self.configs[self.config_index]['max_rules']
        self.config_index = self.config_index


    def get_label_df(self):
        """

        :param df:
        :return:
        """
        label_df = pd.DataFrame()
        # create new label column if original is numerical
        if is_numeric_dtype(self.df[self.class_label]) and self.threshold is not None:

            label_df[self.preprocessed_label_col] = np.where(self.df[self.class_label] >= self.threshold, 1, 0)
            self.pos_class = 1

        else:
            label_df[self.preprocessed_label_col] = self.df[self.class_label]
        return label_df

    def run_pipeline(self):
        """Run the pipeline and return all the interesting metrics"""
        self._fit()
        self.structured_rules = self.create_structured_rules()
        pred = self._get_prediction_labels()
        (true_n, false_p, false_n, true_p, recall_p, recall_n, precision_p, precision_n,) = self._get_confusion_matrix(
            pred)
        avg_True, avg_False = self._get_avg_label_value(pred)
        metric_dict = {"true_n": true_n, "false_p": false_p, "false_n": false_n, "true_p": true_p, "recall_p": recall_p,
            "recall_n": recall_n, "precision_p": precision_p, "precision_n": precision_n, "avg_True": avg_True,
            "avg_False": avg_False, }
        self. metrics = metric_dict

    def _fit(self):
        self.clf = lw.RIPPER(max_rule_conds=self.max_rule_conds, n_discretize_bins=self.n_discretize_bins, k=self.k,
            random_state=self.random_state, max_rules=self.max_rules, )
        self.clf.fit(pd.concat([self.df[self.attrbute_labels], self.label_df], axis=1),
            class_feat=self.preprocessed_label_col, pos_class=self.pos_class, )
        self.rules = self.clf.ruleset_


    def _get_prediction_labels(self):
        pred = self.clf.predict(self.df[self.attrbute_labels])
        return np.array(pred)

    def _get_confusion_matrix(self, pred):
        true_labels = self.label_df[self.preprocessed_label_col].values
        true_n, false_p, false_n, true_p = confusion_matrix(true_labels, pred).ravel()
        recall_p = true_p / (true_p + false_n)
        recall_n = true_n / (true_n + false_p)
        precision_p = true_p / (true_p + false_p)
        precision_n = true_n / (true_n + false_n)
        return (true_n, false_p, false_n, true_p, recall_p, recall_n, precision_p, precision_n,)

    def _get_avg_label_value(self, pred):
        avg_True = self.df[pred][self.class_label].mean()
        avg_False = self.df[~pred][self.class_label].mean()
        return avg_True, avg_False


    def create_structured_rules(self):
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

                conds.append(({'attribute': feature, 'value': str(val), 'unequal_sign': unequal_sign}))
            structured_rules.append(conds)
        return structured_rules

    def _create_attribute_dict(self):
        """For the decision miner algorythm, the names of the attributes are changed a bit. So here a dict is created
        that maps from the changed names to the original names

        :return:
        """
        attr_dict = {}
        for attr in self.attribute_labels:
            changed_str = attr.replace(",", "^").replace("'", "")
            attr_dict[changed_str] = attr
        return attr_dict

    def simplify_rule_config(self):
        """ Reducing the config by 1

        :return:
        """
        if self.config_index > 0:
            self.config_index -= 1
            self.apply_config()
        else:
            raise MinimumValueReachedError()


    def elaborate_rule_config(self):
        """ Reducing the config by 1

        :return:
        """
        if self.config_index < (len(self.configs) - 1):
            self.config_index += 1
            self.apply_config()
        else:
            raise MaximumValueReachedError()
