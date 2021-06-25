from abc import ABCMeta


class BasePersonRecord(metaclass=ABCMeta):
    """
    Base class for all Person record types, whether partial or full.

    Record types support concrete and proxy storage, each with read-write and read-only modes.
    """

    def __init__(self):
        pass
