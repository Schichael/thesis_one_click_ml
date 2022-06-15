from typing import List

import pandas as pd
from pycelonis.celonis_api.pql import pql

from one_click_analysis.process_config.process_config import ActivityTable
from one_click_analysis.process_config.process_config import ProcessConfig


class ViolationProcessor:
    def __init__(
        self,
        conformance_query_str: str,
        process_config: ProcessConfig,
        activity_table_str: str,
        filters: List[pql.PQLFilter],
    ):
        """

        :param conformance_query_str: The query that will return the violation
        :param process_config: ProcessConfig object
        :param activity_table_str: name of the used activity table
        :param filters: The PQL queries that need to be applied.
        """
        self.conformance_query_str = conformance_query_str
        self.process_config = process_config
        self.activity_table_str = activity_table_str
        self.filters = filters

    def get_violations(self):
        pass
        # violations_df = self._get_violations_df()

    def _get_violations_df(self) -> pd.DataFrame:
        """Get the DataFrame with the columns 'case' and 'violations'

        :return: Dataframe with the violations
        """

        activitytable: ActivityTable = self.process_config.table_dict[
            self.activity_table_str
        ]

        pql_query = pql.PQL()
        pql_cases_str = f'"{activitytable.table_str}"."{activitytable.caseid_col_str}"'
        pql_query.add(pql.PQLColumn(query=pql_cases_str, name="case"))
        pql_violations_str = "READABLE(" + self.conformance_query_str + ")"
        pql_query.add(pql.PQLColumn(query=pql_violations_str, name="violations"))
        # Add filters
        pql_query.add(self.filters)
        df = self.process_config.dm.get_data_frame(pql_query)

        return df
