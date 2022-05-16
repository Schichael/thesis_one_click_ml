from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from prediction_builder.data_extraction import ProcessModel
from prediction_builder.data_extraction import ProcessModelFactory
from pycelonis.celonis_api.event_collection.data_model import Datamodel
from pycelonis.celonis_api.event_collection.data_model import DatamodelTable
from pycelonis.celonis_api.pql import pql


class TableColumnType(Enum):
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    DATETIME = "DATETIME"
    TIME = "TIME"
    DATE = "DATE"

    @classmethod
    def categorical_types(cls):
        """Get categorical datatypes"""
        return [cls.STRING, cls.BOOLEAN]

    @classmethod
    def numeric_types(cls):
        """Get numeric datatypes"""
        return [cls.INTEGER, cls.FLOAT]


@dataclass
class TableColumn:
    name: str
    datatype: TableColumnType


@dataclass
class BaseTable:
    table_str: str
    id: str
    columns: List[TableColumn]


@dataclass
class ActivityTable(BaseTable):
    caseid_col_str: str
    activity_col_str: str
    eventtime_col_str: str
    process_model: ProcessModel
    sort_col_str: Optional[str] = None
    case_table_str: Optional[str] = None


@dataclass
class CaseTable(BaseTable):
    caseid_col_str: Optional[str]  # Not sure if needed
    activity_tables_str: List[str]


@dataclass
class OtherTable(BaseTable):
    pass


class ProcessConfig:
    """Holds general configurations of a process model."""

    def __init__(
        self,
        datamodel: Datamodel,
        global_filters: Optional[List[pql.PQLFilter]] = None,
    ):
        """Initialize ProcessConfig class

        :param datamodel: Datamodel
        :param activity_table_str: name of activity table
        :param global_filters: List of PQL filters that are used to get the
        activities.
        """
        # create ProcessModel object
        self.dm = datamodel
        self.global_filters = global_filters
        self.activity_tables = []
        self.case_tables = []
        self.other_tables = []
        self._set_tables()
        # Dictionary mapping table names to table objects
        self.table_dict = self._create_table_dict()

    def _create_table_dict(self) -> Dict[str, BaseTable]:
        """Create dictionary from table name to Table object. This
        method is intended to be called after al activity tables have been
        initialized"""
        tables = self.activity_tables + self.case_tables + self.other_tables
        table_dict = {t.table_str: t for t in tables}
        return table_dict

    def _set_tables(self):
        """Set the table member variables
        :return:
        """

        # Set other activity and case tables
        activity_table_ids = [
            self.dm.data["processConfigurations"][i]["activityTableId"]
            for i in range(len(self.dm.data["processConfigurations"]))
        ]
        for activity_table_id in activity_table_ids:
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
        self, activity_table_str: str
    ) -> Tuple[ActivityTable, CaseTable]:
        """Set the selected activity table and its associated case table

        :param activity_table_str: name of the activity table
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
        activity_table_sort_column = activity_table_process_config["sortingColumn"]
        activity_table_columns = self._create_columns(activity_table)
        activity_table_process_model = ProcessModelFactory.create(
            datamodel=self.dm, activity_table=activity_table_str
        )

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
            process_model=activity_table_process_model,
            sort_col_str=activity_table_sort_column,
            case_table_str=case_table_str,
            columns=activity_table_columns,
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

    def get_activities(self, activity_table_str: str) -> List[str]:
        """Get all activities from an activity table. This is done usong a PQL query.
        :param activity_table_str: name of the activity table
        :return: List with the activities
        """
        activity_table = self.table_dict[activity_table_str]
        q = pql.PQL()
        q += pql.PQLColumn(
            name="Activity",
            query=f""" DISTINCT "{activity_table_str}"."
            {activity_table.activity_col_str}" """,
        )
        q += self.global_filters
        df = self.dm.get_data_frame(q)
        activities = df["Activity"].values.tolist()
        return activities

    def get_categorical_numerical_columns(
        self, table_str: str
    ) -> Tuple[List[str], List[str]]:
        """Get the numerical and categorical columns of a table.

        :param table_str: name of the considered table
        :return: Tuple[categorical_columns, numerical_columns]
        """
        # Get the table with table_str
        table = [
            t
            for t in self.activity_tables + self.case_tables + self.other_tables
            if t.table_str == table_str
        ][0]

        categorical_cols = [
            c.name
            for c in table.columns
            if c.datatype in TableColumnType.categorical_types()
        ]

        numeric_cols = [
            c.name
            for c in table.columns
            if c.datatype in TableColumnType.numeric_types()
        ]

        # If the table is an activity table, need to remove some columns: Activity
        # column, sorting column
        if isinstance(table, ActivityTable):
            if table.activity_col_str in categorical_cols:
                categorical_cols.remove(table.activity_col_str)
            if table.activity_col_str in numeric_cols:
                numeric_cols.remove(table.activity_col_str)

            if table.sort_col_str in categorical_cols:
                categorical_cols.remove(table.sort_col_str)
            if table.sort_col_str in numeric_cols:
                numeric_cols.remove(table.sort_col_str)

        return categorical_cols, numeric_cols
