import abc
from typing import List

from one_click_analysis.feature_processing import attributes


class AttributeSelection(abc.ABC):
    def __init__(
        self,
        selected_attributes: List[attributes.MinorAttribute],
        selected_activity_table_cols: List[str],
        selected_case_table_cols: List[str],
    ):
        self.selected_attributes = selected_attributes
        self.selected_activity_table_cols = selected_activity_table_cols
        self.selected_case_table_cols = selected_case_table_cols

    @abc.abstractmethod
    def update(self):
        pass
