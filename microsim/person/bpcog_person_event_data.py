from dataclasses import dataclass
from typing import List

from microsim.outcome import Outcome


@dataclass
class BPCOGPersonEventData:
    """Dataclass containing events that happened to a single Person during one tick."""
    person_id: int
    mi: List[Outcome]
    stroke: List[Outcome]
    dementia: List[Outcome]
