class OutcomeAssessor:
    DEATH = "death"
    
    def __init__(self, outcomesTypes):
        self.outcomeTypes = outcomesTypes

    def get_outcome(self, person):
        for outcomeType in self.outcomeTypes:
            if outcomeType == OutcomeAssessor.DEATH:
                outcomeDuringSim = person.is_dead()
            else:
                outcomeDuringSim = person.has_outcome_during_simulation(outcomeType)
            
            if outcomeDuringSim:
                return True
        return False  

    def get_name(self):
        name = ""
        for outcome in self.outcomeTypes:
            name += OutcomeAssessor.DEATH if outcome == OutcomeAssessor.DEATH else str(outcome.value) + "-" 
        return name