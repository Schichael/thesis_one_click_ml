from typing import Optional

import pandas as pd
from prediction_builder.data_extraction import ProcessModel
from pycelonis.celonis_api.event_collection.data_model import Datamodel
from pycelonis.celonis_api.pql import pql


def extract_transitions(
    process_model: ProcessModel,
    dm: Datamodel,
    process_start_str: str = "PROCESS_START",
    process_end_str: str = "PROCESS_END",
    is_closed_indicator: Optional[pql.PQLColumn] = None,
):
    """query DataFrame with transition_start_activity, transition_end_activity,
    case_count, transition_count(total number of the transition)

    :param process_model:
    :param dm:
    :param process_start_str:
    :param process_end_str:
    :param is_closed_indicator:
    :return:
    """
    # Validate that process_start_str and process_end_str are not already activity names
    process_start_str_in_activities = process_start_str in process_model.activities
    process_end_str_in_activities = process_end_str in process_model.activities
    if process_start_str_in_activities:
        ValueError("process_start_str must not be a name of an activity.")
    if process_end_str_in_activities:
        ValueError("process_end_str must not be a name of an activity.")

    # Generate is_closed Filter
    is_closed_filter = get_is_closed_filter(is_closed_indicator)

    # First transitions not including transitions form process start
    query_transition_start_str = (
        f'"{process_model.activity_table_str}".'
        f'"{process_model.activity_column_str}"'
    )
    query_transition_start = pql.PQLColumn(
        name="transition_start", query=query_transition_start_str
    )

    query_transition_end_str = (
        f'ACTIVITY_LEAD("{process_model.activity_table_str}".'
        f'"{process_model.activity_column_str}")'
    )
    query_transition_end = pql.PQLColumn(
        name="transition_end", query=query_transition_end_str
    )
    query_case_count_str = f'COUNT_TABLE("{process_model.case_table_str}")'
    query_case_count = pql.PQLColumn(name="case_count", query=query_case_count_str)
    query_transition_count_str = (
        f'COUNT("{process_model.activity_table_str}".'
        f'"{process_model.activity_column_str}")'
    )
    query_transition_count = pql.PQLColumn(
        name="transition_count", query=query_transition_count_str
    )
    pql_query_without_start = pql.PQL()
    pql_query_without_start.add(
        [
            query_transition_start,
            query_transition_end,
            query_case_count,
            query_transition_count,
        ]
    )

    # add filters
    pql_query_without_start.add(process_model.global_filters)
    pql_query_without_start.add(is_closed_filter)

    df_without_start = dm.get_data_frame(pql_query_without_start, chunk_size=10000)
    # Set Process End activity name
    df_without_start.fillna(process_end_str, inplace=True)

    # Now get DataFrame with transitions from process start
    query_transition_end_str = (
        'PU_FIRST("' + process_model.case_table_str + '", '
        '"'
        + process_model.activity_table_str
        + '"."'
        + process_model.activity_column_str
        + '")'
    )
    query_transition_end = pql.PQLColumn(
        name="transition_end", query=query_transition_end_str
    )
    pql_query_start = pql.PQL()
    pql_query_start.add([query_transition_end, query_case_count])

    df_start = dm.get_data_frame(pql_query_start, chunk_size=10000)
    df_start["transition_count"] = df_start["case_count"]
    df_start["transition_start"] = process_start_str

    df = pd.concat([df_without_start, df_start], copy=False, ignore_index=True)

    return df


def get_is_closed_filter(is_closed_query: pql.PQLColumn) -> Optional[pql.PQLFilter]:
    """Returns PQLFilter whether case is closed"""
    if is_closed_query:
        return pql.PQLFilter(f""" FILTER ({is_closed_query.query}) = 1; """)
    else:
        return None
