from dataclasses import dataclass
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
    # The name of the start activity, activity, or transition activities
    specifics: Optional[Union[str, Tuple[str, str]]]
    # Number of cases with violation
    num_cases: int
    # Violation occurrences (generally one case can have the same violation (
    # Activity, Transition) several times)
    num_occurrences: int
    # Metrics: Avg case duration with and without violation, average case steps with
    # and without violation
    metrics: Dict[str, Any]
