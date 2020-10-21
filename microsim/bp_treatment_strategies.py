from microsim.outcome import OutcomeType


class AddASingleBPMedTreatmentStrategy:
    def __init__(self):
        self.sbpLowering = 5.5
        self.dbpLowering = 3.1

    def get_changes_for_person(self, person):
        return {'_antiHypertensiveCount': 1}, {'_bpMedsAdded': 1}, {'_sbp': -1 * self.sbpLowering, '_dbp': -1 * self.dbpLowering}

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def get_treatment_recalibration_for_person(self, person):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def repeat_treatment_strategy(self):
        return False


class AddBPTreatmentMedsToGoal120:
    def __init__(self):
        self.sbpLowering = 5.5
        self.dbpLowering = 3.1

    def get_meds_needed_for_goal(self, person):
        sbpMedCount = int((person._sbp[-1] - 120)/self.sbpLowering)
        dbpMedCount = int((person._dbp[-1] - 80)/self.dbpLowering)
        medCount = dbpMedCount if dbpMedCount < sbpMedCount else sbpMedCount
        return 0 if medCount < 0 else medCount

    def get_changes_for_person(self, person):
        return {'_antiHypertensiveCount': self.get_meds_needed_for_goal(person)}, {'_bpMedsAdded': self.get_meds_needed_for_goal(person)}, {'_sbp': - self.get_meds_needed_for_goal(person) * self.sbpLowering, '_dbp': -self.get_meds_needed_for_goal(person) * self.dbpLowering}

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    def get_treatment_recalibration_for_person(self, person):
        return {OutcomeType.STROKE: 0.79**self.get_meds_needed_for_goal(person), OutcomeType.MI: 0.87**self.get_meds_needed_for_goal(person)}

    def repeat_treatment_strategy(self):
        return True
