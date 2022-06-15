from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union


class ViolationType(Enum):
    """The violation type"""

    START_ACTIVITY = "Start Activity"
    ACTIVITY = "Activity"
    TRANSITION = "Transition"
    INCOMPLETE = "Incomplete"


@dataclass
class Violation:
    """Violation class"""

    violation_type: ViolationType
    # the readable string of the violation in the dataframe
    violation_readable: str
    # The name of the start activity, activity, or transition activities
    specifics: Optional[Union[str, Tuple[str, str]]]
    # Number of cases with violation
    num_cases: Optional[int]
    # Violation occurrences (generally one case can have the same violation (
    # Activity, Transition) several times)
    num_occurrences: Optional[int]
    # Metrics: Avg case duration with and without violation, average case steps with
    # and without violation
    metrics: Dict[str, Any] = field(default_factory=dict)
