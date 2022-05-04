from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import List

from prediction_builder.data_extraction import ProcessModelFactory
from pycelonis.celonis_api.event_collection.data_model import Datamodel


class TableColumnType(Enum):
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    DATETIME = "DATETIME"
    TIME = "TIME"
    DATE = "DATE"


@dataclass
class TableColumn:
    column_name: str
    datatype: TableColumnType


@dataclass
class ActivityTable:
    table_str: str
    caseid_col_str: str
    eventtime_col_str: str
    case_table_str: str
    columns: List = field(default_factory=list)


class ProcessConfig:
    def __init__(self, datamodel: Datamodel, activity_table_str: str):
        # create ProcessModel object
        self.process_model = ProcessModelFactory.create(
            datamodel=datamodel, activity_table=self.activity_table_str
        )

        # Get
