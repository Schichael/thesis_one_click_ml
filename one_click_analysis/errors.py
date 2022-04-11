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
