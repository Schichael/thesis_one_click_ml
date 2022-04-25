from one_click_analysis.feature_processing.attributes.attribute import (
    AttributeDataType,
)


class MinimumValueReachedError(Exception):
    """Raised when a minimum value is already reached and cannot be decreased"""

    def __init__(self):
        message = "The minimum value has been reached and cannot be decreased further"
        super().__init__(message)


class MaximumValueReachedError(Exception):
    """Raised when a minimum value is already reached and cannot be decreased"""

    def __init__(self):
        message = "The maximum value has been reached and cannot be increased further"
        super().__init__(message)


class NotAValidAttributeError(Exception):
    """Raised when an attribute is not valid or does not exist"""

    def __init__(self, attr):
        message = str(attr) + " is not avalid attribute."
        super().__init__(message)


class DecisionRuleNotValidLabelTypesError(Exception):
    def __init__(self, labels):
        label_datatypes = [label.attribute_data_type for label in labels]
        message = (
            f"Decison rules require all categorical labels or exactly one "
            f"numerical label without any other labels. But got: {label_datatypes}"
        )
        super().__init__(message)


class ConfiguratorNotSetError(Exception):
    def __init__(self):
        message = (
            f"A configuration must have a configurator initialized. This is done "
            f"by initializing a Configurator object with this Configuration "
            f"instance"
        )
        super().__init__(message)


class WrongFeatureTypeError(Exception):
    def __init__(self, feature_col_name: str, datatype: AttributeDataType):
        message = (
            f"Feature type mus be either categorical or numerical but type of "
            f"feature '{feature_col_name}' is: {datatype}"
        )
        super().__init__(message)
