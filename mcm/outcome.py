from enum import Enum


class Outcome:
    def __init__(self, type, fatal, **kwargs):
        self.type = type
        self.fatal = fatal
        self.properties = {**kwargs}

    def __repr__(self):
        return (f"Outcome type: {self.type}, fatal: {self.fatal}")


class OutcomeType(Enum):
    STROKE = "stroke"
    MI = "mi"
