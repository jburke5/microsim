from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType
from microsim.treatment_strategies.bp_treatment_strategies import (
    AddNBPMedsTreatmentStrategy,
    AddBPTreatmentMedsToGoal120,
    SprintTreatment,
)


class TreatmentStrategyRepository:
    def __init__(self):
        self._repository = dict()
        for ts in TreatmentStrategiesType:
            self._repository[ts.value] = None

    @classmethod
    def from_string(cls, name):
        '''Build a repository from a shorthand name for a common BP strategy.
        Recognized names: "1bpMedsAdded".."4bpMedsAdded", "toGoal120", "sprint",
        "noTreatment" (empty repository).'''
        repo = cls()
        if name == "1bpMedsAdded":
            repo._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(1)
        elif name == "2bpMedsAdded":
            repo._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(2)
        elif name == "3bpMedsAdded":
            repo._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(3)
        elif name == "4bpMedsAdded":
            repo._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(4)
        elif name == "toGoal120":
            repo._repository[TreatmentStrategiesType.BP.value] = AddBPTreatmentMedsToGoal120()
        elif name == "noTreatment":
            pass
        elif name == "sprint":
            repo._repository[TreatmentStrategiesType.BP.value] = SprintTreatment()
        else:
            raise ValueError(f"Unrecognized treatment strategy shorthand: {name!r}")
        return repo
