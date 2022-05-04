from typing import List
from typing import Optional

from pycelonis import get_celonis
from pycelonis.celonis_api.event_collection.data_model import Datamodel
from pycelonis.celonis_api.event_collection.data_model import DatamodelTable


def get_dm(datamodel_name: str, celonis_login: Optional[dict] = None):
    """Get datamodel from Celonis

    :param datamodel_name: name or id of datamodel
    :param celonis_login: login data for Celonis
    :return:
    """
    if celonis_login is None:
        celonis = get_celonis()
    else:
        celonis = get_celonis(**celonis_login)
    dm = celonis.datamodels.find(datamodel_name)
    return dm


def get_activity_tables(dm: Datamodel) -> List[DatamodelTable]:
    """Get activity tables as a list of DatamodelTable.

    :param dm: the datamodel
    :return: List of activity tables
    """
    activity_table_ids = [
        dm.data["processConfigurations"][i]["activityTableId"]
        for i in range(len(dm.data["processConfigurations"]))
    ]
    activity_tables = [t for t in dm.tables if t.id in activity_table_ids]
    return activity_tables
