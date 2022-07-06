from typing import Dict, Any, List

from pycelonis.celonis_api.pql import pql


class Configurator:
    """Class that holds the configurations"""

    def __init__(self):
        # Dictionary that holds configurations. It is structured as follows:
        # {"configuration_identifier_str": "some_config_key": config_value}
        self.config_dict: Dict[str : Dict[str:Any]] = {}
        # Dictionary that holds filters. It is structured as follows:
        # {"configuration_identifier_str": [PQLFilter1, PQLFilter2, ...]
        self.filter_dict: Dict[str : List[pql.PQLFilter]] = {}

    def get_all_filters(self) -> List[pql.PQLFilter]:
        """Get all filters stored in filter_dict as a list"""
        all_filters = []
        for filters in self.filter_dict.values():
            all_filters.append(filters)
        return all_filters
