from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple

from pycelonis.celonis_api.event_collection.data_model import Datamodel
from pycelonis.celonis_api.event_collection.data_model import DatamodelTable
from pycelonis.celonis_api.pql import pql

# from prediction_builder.data_extraction import ProcessModelFactory


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
    activity_col_str: str
    eventtime_col_str: str
    id: str
    case_table_str: Optional[str] = None
    columns: List[TableColumn] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)


@dataclass
class CaseTable:
    table_str: str
    caseid_col_str: Optional[str]  # Not sure if needed
    activity_tables_str: List[str]
    id: str
    columns: List[TableColumn] = field(default_factory=list)


@dataclass
class OtherTable:
    table_str: str
    id: str
    columns: List[TableColumn] = field(default_factory=list)


class ProcessConfig:
    """Holds general configurations of a process model."""

    def __init__(
        self,
        datamodel: Datamodel,
        activity_table_str: str,
        global_filters: Optional[List[pql.PQLFilter]] = None,
    ):
        """Initialize ProcessConfig class

        :param datamodel: Datamodel
        :param activity_table_str: name of activity table
        :param global_filters: List of PQL filters that are used to get the
        activities.
        """
        # create ProcessModel object
        # self.process_model = ProcessModelFactory.create(
        #    datamodel=datamodel, activity_table=activity_table_str
        # )
        self.dm = datamodel
        self.global_filters = global_filters
        self.primary_activity_table = None
        self.activity_tables = []
        self.primary_case_table = None
        self.case_tables = []
        self.other_tables = []
        self._set_tables(activity_table_str)

    def _set_tables(self, activity_table_str):
        """Set the table member variables

        :param activity_table_str: name of the primary activity table
        :return:
        """
        # Set activity and case table
        (
            self.primary_activity_table,
            self.primary_case_table,
        ) = self._set_activity_case_table(
            activity_table_str, is_primary_activity_table=True
        )
        self.activity_tables.append(self.primary_activity_table)
        if self.primary_case_table is not None:
            self.case_tables.append(self.primary_case_table)

        # Set other activity and case tables
        activity_table_ids = [
            self.dm.data["processConfigurations"][i]["activityTableId"]
            for i in range(len(self.dm.data["processConfigurations"]))
        ]
        for activity_table_id in activity_table_ids:
            if self.primary_activity_table.id == activity_table_id:
                continue
            activity_table = self.dm.tables.find(activity_table_id)
            activity_table_str = activity_table.name
            activity_table, case_table = self._set_activity_case_table(
                activity_table_str
            )
            self.activity_tables.append(activity_table)
            if case_table is not None:
                self.case_tables.append(case_table)

        # Set other tables
        already_selected_table_ids = []
        for case_table in self.case_tables:
            already_selected_table_ids.append(case_table.id)
        for activity_table in self.activity_tables:
            already_selected_table_ids.append(activity_table.id)

        for table in self.dm.tables:
            if table.id not in already_selected_table_ids:
                other_table = self._gen_other_table(table)
                self.other_tables.append(other_table)

    def _set_activity_case_table(
        self, activity_table_str: str, is_primary_activity_table: bool = False
    ) -> Tuple[ActivityTable, CaseTable]:
        """Set the selected activity table and its associated case table

        :param activity_table_str: name of the activity table
        :param is_primary_activity_table: whether it's the primary activity table. If
        true, activities are set, else not.
        :return: ActivityTable object and CaseTable object
        """
        activity_table = self.dm.tables.find(activity_table_str)
        activity_table_id = activity_table.id
        activity_table_process_config = [
            el
            for el in self.dm.data["processConfigurations"]
            if el["activityTableId"] == activity_table_id
        ][0]
        case_table_id = activity_table_process_config["caseTableId"]

        activity_table_case_id_column = activity_table_process_config["caseIdColumn"]
        activity_table_activity_column = activity_table_process_config["activityColumn"]
        activity_table_eventtime_column = activity_table_process_config[
            "timestampColumn"
        ]
        activity_table_columns = self._create_columns(activity_table)
        if is_primary_activity_table:
            activity_table_activities = self._get_activities(
                activity_table_str, activity_table_activity_column
            )
        else:
            activity_table_activities = []

        if case_table_id:
            # Check if case table object already exists
            case_table_old = self._get_case_table(case_table_id)
            if case_table_old is not None:
                case_table_old.activity_tables_str.append(activity_table_str)
                case_table_obj = None
                case_table_str = case_table_old.table_str
            else:
                case_table = self.dm.tables.find(case_table_id)
                case_table_obj = self._gen_case_table(
                    case_table=case_table,
                    activity_table_str=activity_table_str,
                    activity_table_id=activity_table_id,
                )
                case_table_str = case_table_obj.table_str
        else:
            case_table_obj = None
            case_table_str = None

        activity_table_obj = ActivityTable(
            table_str=activity_table_str,
            caseid_col_str=activity_table_case_id_column,
            activity_col_str=activity_table_activity_column,
            eventtime_col_str=activity_table_eventtime_column,
            id=activity_table_id,
            case_table_str=case_table_str,
            columns=activity_table_columns,
            activities=activity_table_activities,
        )

        return activity_table_obj, case_table_obj

    def _get_case_table(self, case_table_id: str) -> CaseTable or None:
        """Get a case table from the table id

        :param case_table_id: id of the case table
        :return: CaseTable object if case table exists already in self.case_tables,
        else None
        """
        for case_table in self.case_tables:
            if case_table.id == case_table_id:
                return case_table
        return None

    def _gen_case_table(
        self, case_table: DatamodelTable, activity_table_str, activity_table_id
    ) -> CaseTable:
        """Generate a CaseTable object.

        :param case_table: DataModel object of the case table
        :param activity_table_str: name of the activity table
        :param activity_table_id: id of the activity table
        :return: CaseTable object
        """
        case_table_str = case_table.name
        case_table_id = case_table.id
        foreign_key_case_id = next(
            (
                item
                for item in self.dm.data["foreignKeys"]
                if item["sourceTableId"] == case_table_id
                and item["targetTableId"] == activity_table_id
            ),
            None,
        )
        # It can be that the activity table and the associated case table are not
        # connected via a foreign key. Therefore, it can happen that
        # foreign_key_case_id is None.
        if foreign_key_case_id is not None:
            case_case_id = foreign_key_case_id["columns"][0]["sourceColumnName"]
        else:
            case_case_id = None
        case_table_columns = self._create_columns(case_table)
        case_table_obj = CaseTable(
            table_str=case_table_str,
            caseid_col_str=case_case_id,
            activity_tables_str=[activity_table_str],
            id=case_table_id,
            columns=case_table_columns,
        )
        return case_table_obj

    def _gen_other_table(self, table: DatamodelTable) -> OtherTable:
        """Generate OtherTable object from DatamodelTable

        :param table: Datamodel table object
        :return: OtherTable object
        """
        table_str = table.name
        table_id = table.id
        table_columns = self._create_columns(table)
        other_table = OtherTable(
            table_str=table_str, id=table_id, columns=table_columns
        )
        return other_table

    def _create_columns(self, table: DatamodelTable) -> List[TableColumn]:
        """Create list of the columns of the input table as TableColumn objects

        :param table: input table
        :return: List of TableColumn objects
        """
        cols = [
            TableColumn(c["name"], self._gen_column_datatype(c["type"]))
            for c in table.columns
        ]
        return cols

    def _gen_column_datatype(self, datatype_str: str) -> TableColumnType:
        """Generate the TableColumnType from the string of the 'type' value
        of an entry from DatamodelTable.columns

        :param datatype_str: 'type' value of an entry from DatamodelTable.columns
        :return: TableColumnType
        """
        datatype_mapping = {
            "STRING": TableColumnType.STRING,
            "BOOLEAN": TableColumnType.BOOLEAN,
            "INTEGER": TableColumnType.INTEGER,
            "FLOAT": TableColumnType.FLOAT,
            "DATETIME": TableColumnType.DATETIME,
            "TIME": TableColumnType.TIME,
            "DATE": TableColumnType.DATE,
        }
        return datatype_mapping[datatype_str]

    def _get_activities(
        self,
        activity_table_str: str,
        activity_column_str: str,
    ):
        q = pql.PQL()
        q += pql.PQLColumn(
            name="Activity",
            query=f""" DISTINCT "{activity_table_str}"."{activity_column_str}" """,
        )
        q += self.global_filters
        df = self.dm.get_data_frame(q)
        activities = df["Activity"].values.tolist()
        return activities
