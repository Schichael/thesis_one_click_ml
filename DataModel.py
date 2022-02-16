class EmptyTable:
    def __init__(self):
        self.columns = []

    def __bool__(self):
        return False


class DataModelInfo:
    def __init__(self, dm):
        self.dm = dm
        self.activity_table_name = None
        self.case_table_name = None
        self.activity_table = None
        self.case_table = None
        self.case_case_key = None
        self.activity_case_key = None
        self.eventtime_col = ""
        self.sort_col = ""
        self.categorical_types = ["STRING", "BOOLEAN"]
        self.numerical_types = ["INTEGER", "FLOAT"]
        self.static_numerical_cols = []
        self.static_categorical_cols = []
        self.dynamic_numerical_cols = []
        self.dynamic_categorical_cols = []

        self._init_datamodel(dm)

    def _init_datamodel(self, dm):
        """Initialize datamodel parameters

        :param dm: input Datamodel
        :return:
        """
        # get activity and case table IDs
        activity_table_id = dm.data["processConfigurations"][0]["activityTableId"]
        case_table_id = dm.data["processConfigurations"][0]["caseTableId"]
        self.activity_table = dm.tables.find(activity_table_id)
        self.eventtime_col = dm.data["processConfigurations"][0]["timestampColumn"]
        self.sort_col = dm.data["processConfigurations"][0]["sortingColumn"]
        self.activity_col = dm.data["processConfigurations"][0]["activityColumn"]
        self.activity_table_name = self.activity_table.name

        if case_table_id:
            self.case_table = dm.tables.find(case_table_id)

            foreign_key_case_id = next(
                (
                    item
                    for item in dm.data["foreignKeys"]
                    if item["sourceTableId"] == case_table_id
                       and item["targetTableId"] == activity_table_id
                ),
                None,
            )

            self.activity_case_key = foreign_key_case_id["columns"][0]["targetColumnName"]
            self.case_case_key = foreign_key_case_id["columns"][0]["sourceColumnName"]
            self.case_table_name = self.case_table.name
            self._set_dynamic_features_PQL()
            self._set_static_features_PQL()
        else:
            self.case_table = EmptyTable()
            self.case_case_key = ''
            self.case_table_name = ''
            self.activity_case_key = dm.data["processConfigurations"][0]['caseIdColumn']
            self._set_dynamic_features_PQL()

    def _set_static_features_PQL(self):
        for attribute in self.case_table.columns:
            if attribute['type'] in self.categorical_types and attribute['name'] not in [self.case_case_key,
                                                                                         self.sort_col]:
                self.static_categorical_cols.append(attribute['name'])
            elif attribute['type'] in self.numerical_types and attribute['name'] not in [self.case_case_key,
                                                                                         self.sort_col]:
                self.static_numerical_cols.append(attribute['name'])

    def _set_dynamic_features_PQL(self):
        for attribute in self.activity_table.columns:
            if attribute['type'] in self.categorical_types and attribute['name'] not in [self.activity_case_key,
                                                                                         self.sort_col]:
                self.dynamic_categorical_cols.append(attribute['name'])
            elif attribute['type'] in self.numerical_types and attribute['name'] not in [self.activity_case_key,
                                                                                         self.sort_col]:
                self.dynamic_numerical_cols.append(attribute['name'])
