from microsim.outcome import OutcomeType
from abc import ABC


class BaseTreatmentStrategy(ABC):
    def get_changes_vectorized(self, x):
        raise RuntimeError("Vectorized Treatment not implemented for this strategy")


class AddASingleBPMedTreatmentStrategy(BaseTreatmentStrategy):
    sbpLowering = 5.5
    dbpLowering = 3.1

    def __init__(self):
        pass

    def get_changes_for_person(self, person):
        return (
            {"_antiHypertensiveCount": 1},
            {"_bpMedsAdded": 1},
            {
                "_sbp": -1 * AddASingleBPMedTreatmentStrategy.sbpLowering,
                "_dbp": -1 * AddASingleBPMedTreatmentStrategy.dbpLowering,
            },
        )

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def get_treatment_recalibration_for_person(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def repeat_treatment_strategy(self):
        return False

    def get_changes_vectorized(self, x):
        x.antiHypertensiveCountNext = x.antiHypertensiveCountNext + 1
        x.bpMedsAddedNext = 1
        x.sbpNext = x.sbpNext - AddASingleBPMedTreatmentStrategy.sbpLowering
        x.dbpNext = x.dbpNext - AddASingleBPMedTreatmentStrategy.dbpLowering
        return x

    def rollback_changes_vectorized(self, x):
        x.antiHypertensiveCountNext = x.antiHypertensiveCountNext - 1
        x.sbpNext = x.sbpNext + AddASingleBPMedTreatmentStrategy.sbpLowering
        x.dbpNext = x.dbpNext + AddASingleBPMedTreatmentStrategy.dbpLowering
        x.bpMedsAddedNext = 0
        return x


class AddBPTreatmentMedsToGoal120(BaseTreatmentStrategy):
    sbpLowering = 5.5
    dbpLowering = 3.1
    sbpGoal = 120
    dbpGoal = 65

    def __init__(self):
        pass

    def getGoal(self, person):
        return {'sbp' : 120, 'dbp' : 65}
    
    def get_meds_needed_for_goal(self, sbp, dbp, goal):
        sbpMedCount = int((sbp - goal['sbp']) / AddBPTreatmentMedsToGoal120.sbpLowering)
        dbpMedCount = int((dbp - goal['dbp']) / AddBPTreatmentMedsToGoal120.dbpLowering)
        medCount = dbpMedCount if dbpMedCount < sbpMedCount else sbpMedCount
        return 0 if medCount < 0 else medCount

    def get_changes_for_person(self, person):
        medsForGoal = self.get_meds_needed_for_goal(person._sbp[-1], person._dbp[-1], self.getGoal(person))
        return (
            {"_antiHypertensiveCount": medsForGoal},
            {"_bpMedsAdded": medsForGoal},
            {
                "_sbp": -1 * medsForGoal * AddBPTreatmentMedsToGoal120.sbpLowering,
                "_dbp": -1 * medsForGoal * AddBPTreatmentMedsToGoal120.dbpLowering,
            },
        )

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def get_treatment_recalibration_for_person(self, person):
        medsForGoal = self.get_meds_needed_for_goal(person.sbp[-1], person.dbp[-1])
        return {OutcomeType.STROKE: 0.79 ** medsForGoal, OutcomeType.MI: 0.87 ** medsForGoal}

    def repeat_treatment_strategy(self):
        return True

    def get_changes_vectorized(self, x):
        medsNeeded = self.get_meds_needed_for_goal(x.sbpNext, x.dbpNext)
        x.bpMedsAddedNext = medsNeeded
        x.antiHypertensiveCountNext = x.antiHypertensiveCountNext + medsNeeded
        x.sbpNext = x.sbpNext - medsNeeded * AddBPTreatmentMedsToGoal120.sbpLowering
        x.dbpNext = x.dbpNext - medsNeeded * AddBPTreatmentMedsToGoal120.dbpLowering
        return x

    def rollback_changes_vectorized(self, x):
        x.antiHypertensiveCountNext = x.antiHypertensiveCountNext - x.bpMedsAddedNext
        x.sbpNext = x.sbpNext + x.bpMedsAddedNext * AddBPTreatmentMedsToGoal120.sbpLowering
        x.dbpNext = x.dbpNext + x.bpMedsAddedNext * AddBPTreatmentMedsToGoal120.dbpLowering
        x.bpMedsAddedNext = 0
        return x


class NoBPTreatmentNoBPChange(BaseTreatmentStrategy):
    def __init__(self):
        pass

    def get_changes_for_person(self, person):
        current = person._antiHypertensiveCount[-1]
        changeSBP = person._sbp[-1] - person._sbp[-2]
        changeDBP = person._dbp[-1] - person._dbp[-2]
        return (
            {"_antiHypertensiveCount": -1 * current},
            {"_bpMedsAdded": -1 * current},
            {"_sbp": -1 * changeSBP, "_dbp": -1 * changeDBP},
        )

    def get_treatment_recalibration_for_population(self):
        return None

    def get_treatment_recalibration_for_person(self):
        return None

    def repeat_treatment_strategy(self):
        return True

    def get_changes_vectorized(self, x):
        x.bpMedsAddedNext = 0
        x.antiHypertensiveCountNext = 0
        x.sbpNext = x.sbp
        x.dbpNext = x.dbp
        return x

    def rollback_changes_vectorized(self, x):
        return x
