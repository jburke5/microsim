from enum import IntEnum


class RaceEthnicity(IntEnum):
    """
    NHANES Race Ethinity enumeration.

    Follows RIDRETH1.
    """

    MEXICAN_AMERICAN = 1
    OTHER_HISPANIC = 2
    NON_HISPANIC_WHITE = 3
    NON_HISPANIC_BLACK = 4
    OTHER = 5
    ASIAN = 6

