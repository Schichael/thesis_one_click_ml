from typing import List
from typing import Optional

import pandas as pd
from pycelonis.celonis_api.pql import pql

from one_click_analysis.process_config.process_config import ActivityTable
from one_click_analysis.process_config.process_config import ProcessConfig
from one_click_analysis.violation_processing.violation import Violation
from one_click_analysis.violation_processing.violation import ViolationType


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

    def get_violations(self) -> List[Violation]:
        """Get Violations from the process.

        :return: List with Violation objects
        """
        violations_df = self._get_violations_df()
        unique_violations = violations_df["violation"].unique()
        value_counts = violations_df["violation"].value_counts()
        violations = []
        for violation in unique_violations:
            violation = self._create_violation_object(
                violation, violations_df, value_counts
            )
            if violation is not None:
                violations.append(violation)
        return violations

    def _create_violation_object(
        self, violation_str: str, violations_df: pd.DataFrame, value_counts: pd.Series
    ) -> Optional[Violation]:
        """Create a Violation object from a violation string.

        :param violation_str: the violation string
        :param violations_df: the DataFrame with the violations
        :return: Violation object if it is a supported violation. Else None
        """
        # Number of cases with violation
        num_cases = len(
            violations_df[violations_df["violation"] == violation_str]["case"].unique()
        )
        # Number of violation occurrences
        num_occurrences = value_counts[violation_str]
        # Check if violation is a violating Start activity
        if violation_str.endswith(" executed as start activity"):
            violation_type = ViolationType.START_ACTIVITY
            specifics = violation_str[: -len(" executed as start activity")]
        elif violation_str.endswith(" is an undesired activity"):
            violation_type = ViolationType.ACTIVITY
            specifics = violation_str[: -len(" is an undesired activity")]
        elif " is followed by " in violation_str:
            violation_type = ViolationType.TRANSITION
            specifics = tuple(violation_str.split(" is followed by "))
        elif violation_str == "Incomplete":
            violation_type = ViolationType.INCOMPLETE
            specifics = None
        else:
            return None

        return Violation(
            violation_type=violation_type,
            specifics=specifics,
            num_cases=num_cases,
            num_occurrences=num_occurrences,
        )

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
        pql_query.add(pql.PQLColumn(query=pql_violations_str, name="violation"))
        # Add filters
        pql_query.add(self.filters)
        df = self.process_config.dm.get_data_frame(pql_query)

        return df
