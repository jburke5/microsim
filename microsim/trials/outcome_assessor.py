class OutcomeAssessor:
    def __init__(self, outcomesTypes):
        self.outcomeTypes = outcomesTypes

    def get_outcome(self, person):
        for outcomeType in self.outcomeTypes:
            outcomeDuringSim = person.has_outcome_during_simulation (outcomeType)
            if outcomeDuringSim:
                return True
        return False   