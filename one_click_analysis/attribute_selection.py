import abc
from typing import List

from one_click_analysis.feature_processing import attributes
from one_click_analysis.feature_processing.attributes_new.feature import Feature


class AttributeSelection(abc.ABC):
    """This class is used to define the behaviour when  attribute selection is
    updated. The update() method needs to be overwritten

    """

    def __init__(
        self,
        selected_attributes: List[attributes.MinorAttribute],
        selected_activity_table_cols: List[str],
        selected_case_table_cols: List[str],
        features: List[Feature],
    ):
        self.selected_attributes = selected_attributes
        self.selected_activity_table_cols = selected_activity_table_cols
        self.selected_case_table_cols = selected_case_table_cols
        self.features = features

    @abc.abstractmethod
    def update(self):
        """Define what shall happen when attribute selection is updated.

        :return:
        """
        pass
