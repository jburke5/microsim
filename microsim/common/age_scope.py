from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AgeScope:
    """A subset of ages used for prevalence/incidence pooling and calibration.

    lo/hi are inclusive bounds; None means unbounded on that side.
        AgeScope()              -> all ages
        AgeScope(65, None)      -> age >= 65
        AgeScope(70, 74)        -> 70..74 inclusive
        AgeScope(75, 75)        -> exactly 75
    """

    lo: Optional[int] = None
    hi: Optional[int] = None

    def __post_init__(self):
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f"AgeScope lo>hi: {self.lo}>{self.hi}")

    def contains(self, age: int) -> bool:
        return (self.lo is None or age >= self.lo) and (self.hi is None or age <= self.hi)

    @property
    def label(self) -> str:
        if self.lo is None and self.hi is None:
            return "pooled_overall"
        if self.lo == self.hi:
            return f"age_{self.lo}"
        if self.hi is None:
            return f"pooled_{self.lo}_plus"
        if self.lo is None:
            return f"pooled_up_to_{self.hi}"
        return f"age_group_{self.lo}-{self.hi}"
