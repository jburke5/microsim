from enum import Enum


class Outcome:
    def __init__(self, type, fatal, **kwargs):
        self.type = type
        self.fatal = fatal
        self.properties = {**kwargs}


class OutcomeType(Enum):
    STROKE = "stroke"
    MI = "mi"
