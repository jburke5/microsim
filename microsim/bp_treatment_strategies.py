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
        return {'_antiHypertensiveCount': 1}, {'_bpMedsAdded': 1}, {'_sbp': -1 * AddASingleBPMedTreatmentStrategy.sbpLowering, '_dbp': -1 * AddASingleBPMedTreatmentStrategy.dbpLowering}

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def get_treatment_recalibration_for_person(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def repeat_treatment_strategy(self):
        return False

    def get_changes_vectorized(self, x):
        x.antiHypertensiveCountNext = x.antiHypertensiveCountNext + 1
        x.sbpNext = x.sbpNext - AddASingleBPMedTreatmentStrategy.sbpLowering
        x.dbpNext = x.dbpNext - AddASingleBPMedTreatmentStrategy.dbpLowering
        return x


class AddBPTreatmentMedsToGoal120(BaseTreatmentStrategy):
    sbpLowering = 5.5
    dbpLowering = 3.1

    def __init__(self):
        pass

    def get_meds_needed_for_goal(self, sbp, dbp):
        sbpMedCount = int((sbp - 120)/AddBPTreatmentMedsToGoal120.sbpLowering)
        dbpMedCount = int((dbp - 65)/AddBPTreatmentMedsToGoal120.dbpLowering)
        medCount = dbpMedCount if dbpMedCount < sbpMedCount else sbpMedCount
        return 0 if medCount < 0 else medCount

    def get_changes_for_person(self, person):
        medsForGoal = self.get_meds_needed_for_goal(person.sbp[-1], person.dbp[-1])
        return {'_antiHypertensiveCount': medsForGoal}, {'_bpMedsAdded': medsForGoal}, {'_sbp': -1 * medsForGoal * AddBPTreatmentMedsToGoal120.sbpLowering, '_dbp': -1 * medsForGoal * AddBPTreatmentMedsToGoal120.dbpLowering}

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def get_treatment_recalibration_for_person(self, person):
        medsForGoal = self.get_meds_needed_for_goal(person.sbp[-1], person.dbp[-1])
        return {OutcomeType.STROKE: 0.79**medsForGoal, OutcomeType.MI: 0.87**medsForGoal}

    def repeat_treatment_strategy(self):
        return True

    def get_changes_vectorized(self, x):
        medsNeeded = self.get_meds_needed_for_goal(x.sbpNext, x.dbpNext)
        x.antiHypertensiveCountNext = x.antiHypertensiveCountNext + medsNeeded
        x.sbpNext = x.sbpNext - medsNeeded * AddBPTreatmentMedsToGoal120.sbpLowering
        x.dbpNext = x.dbpNext - medsNeeded * AddBPTreatmentMedsToGoal120.dbpLowering
        return x
