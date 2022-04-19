from typing import List

import pandas as pd

from one_click_analysis.feature_processing.attributes_new.attribute import Attribute


class PostProcessor:
    """Post processes a dataframe and generates a list of Feature objects.

    """

    def __init__(self, df: pd.DataFrame, attributes: List[Attribute]):
        self.df = df
        self.attributes = attributes

