from dataclasses import dataclass
from typing import List

from microsim.outcome import Outcome


@dataclass
class BPCOGPersonEventData:
    """Dataclass containing events that happened to a single Person during one tick."""

    mi: Outcome
    stroke: Outcome
    dementia: Outcome
