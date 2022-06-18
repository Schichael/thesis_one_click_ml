from typing import List
from typing import Optional

import pandas as pd
from pycelonis.celonis_api.pql import pql

from one_click_analysis.feature_processing.attributes.static_attributes import (
    CaseDurationAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    EventCountAttribute,
)
from one_click_analysis.feature_processing.attributes.static_attributes import (
    StartActivityTimeAttribute,
)
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
        is_closed_query_str: str,
    ):
        """

        :param conformance_query_str: The query that will return the violation
        :param process_config: ProcessConfig object
        :param activity_table_str: name of the used activity table
        :param filters: The PQL queries that need to be applied without is_closed
        :param is_closed_query_str: is_closed query string
        filter.
        """
        self.conformance_query_str = conformance_query_str
        self.process_config = process_config
        self.activity_table_str = activity_table_str
        self.filters = filters
        self.is_closed_query_str = is_closed_query_str
        self.violations_df = self._get_violations_df()
        self.violations = self._create_violations()

    def _create_violations(self) -> List[Violation]:
        """Get Violations from the process.

        :return: List with Violation objects
        """
        unique_violations = self.violations_df["violation"].unique()
        violations = []
        for violation in unique_violations:
            violation = self._create_violation_object(violation, self.violations_df)
            if violation is not None:
                violations.append(violation)
        return violations

    def _create_violation_object(
        self,
        violation_str: str,
        violations_df: pd.DataFrame,
    ) -> Optional[Violation]:
        """Create a Violation object from a violation string.

        :param violation_str: the violation string
        :param violations_df: the DataFrame with the violations
        :return: Violation object if it is a supported violation. Else None
        """
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

        # If the violation is of type activity or transition, only use the closed cases
        if violation_type in [ViolationType.ACTIVITY, ViolationType.TRANSITION]:
            violations_df = violations_df[violations_df["IS_CLOSED"] == 1]

        # All cases that have this violation
        avg_case_durations_all = (
            violations_df.groupby("case").first()["case " "duration"].mean()
        )
        avg_events_all = (
            violations_df.groupby("case").count()["case " "duration"].mean()
        )
        cases_violation = violations_df[violations_df["violation"] == violation_str][
            "case"
        ].unique()
        curr_violation_df = violations_df[violations_df["case"].isin(cases_violation)]
        # All cases that don't have this violation
        cases_violation = curr_violation_df["case"].unique()
        curr_not_violation_df = violations_df[
            ~violations_df["case"].isin(cases_violation)
        ]
        # Number of cases with violation
        num_cases = len(curr_violation_df["case"].unique())
        # Number of violation occurrences
        num_occurrences = len(curr_violation_df)

        # Case duration with violation
        curr_violation_df_grouped = curr_violation_df.groupby("case").first()
        avg_case_duration_with_violation = curr_violation_df_grouped[
            "case duration"
        ].mean()
        # Case duration without violation
        curr_not_violation_df_grouped = curr_not_violation_df.groupby("case").first()
        avg_case_duration_without_violation = curr_not_violation_df_grouped[
            "case " "duration"
        ].mean()

        # Number of events with violation
        curr_violation_df_grouped = curr_violation_df.groupby("case").count()
        avg_num_events_with_violation = curr_violation_df_grouped[
            "case duration"
        ].mean()

        # Number of events without violation
        curr_not_violation_df_grouped = curr_not_violation_df.groupby("case").count()
        avg_num_events_without_violation = curr_not_violation_df_grouped[
            "case " "duration"
        ].mean()

        effect_on_case_duration = (
            avg_case_duration_with_violation - avg_case_durations_all
        )
        effect_on_event_count = avg_num_events_with_violation - avg_events_all

        metrics = {
            "effect_on_case_duration": effect_on_case_duration,
            "effect_on_event_count": effect_on_event_count,
            "avg_case_duration_with_violation": avg_case_duration_with_violation,
            "avg_case_duration_without_violation": avg_case_duration_without_violation,
            "avg_num_events_with_violation": avg_num_events_with_violation,
            "avg_num_events_without_violation": avg_num_events_without_violation,
        }
        return Violation(
            violation_type=violation_type,
            violation_readable=violation_str,
            specifics=specifics,
            num_cases=num_cases,
            num_occurrences=num_occurrences,
            metrics=metrics,
        )

    def _get_violations_df(self) -> pd.DataFrame:
        """Get the DataFrame with the columns 'case' and 'violations'

        :return: Dataframe with the violations
        """

        activitytable: ActivityTable = self.process_config.table_dict[
            self.activity_table_str
        ]

        pql_query = pql.PQL()
        # is_closed
        pql_query.add(pql.PQLColumn(query=self.is_closed_query_str, name="IS_CLOSED"))
        pql_cases_str = f'"{activitytable.table_str}"."{activitytable.caseid_col_str}"'
        pql_query.add(pql.PQLColumn(query=pql_cases_str, name="case"))
        pql_violations_str = "READABLE(" + self.conformance_query_str + ")"
        pql_query.add(pql.PQLColumn(query=pql_violations_str, name="violation"))
        case_duration_attr = CaseDurationAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
            time_aggregation="SECONDS",
        )
        pql_case_duration_str = case_duration_attr.pql_query.query
        pql_query.add(
            pql.PQLColumn(query=pql_case_duration_str, name="case " "duration")
        )
        pql_start_activity_time_attr = StartActivityTimeAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
        )
        pql_start_activity_time_str = pql_start_activity_time_attr.pql_query.query
        pql_query.add(
            pql.PQLColumn(query=pql_start_activity_time_str, name="Start activity time")
        )
        event_count_attr = EventCountAttribute(
            process_config=self.process_config,
            activity_table_str=self.activity_table_str,
        )
        pql_event_count_str = event_count_attr.pql_query.query
        pql_query.add(pql.PQLColumn(query=pql_event_count_str, name="event count"))
        # Add filters
        pql_query.add(self.filters)
        df = self.process_config.dm.get_data_frame(pql_query)

        return df
