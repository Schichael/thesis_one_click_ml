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
