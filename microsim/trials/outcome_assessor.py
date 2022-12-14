class OutcomeAssessor:
    DEATH = "death"
    
    def __init__(self, outcomeTypes):
        self.outcomeTypes = outcomeTypes #assumed to be a python list

    def get_outcome(self, person): #RegressionAnalysis assumes that this method returns 0 or 1
        for outcomeType in self.outcomeTypes:
            if outcomeType == OutcomeAssessor.DEATH:
                outcomeDuringSim = person.is_dead()
            else:
                outcomeDuringSim = person.has_outcome_during_simulation(outcomeType)
            
            if outcomeDuringSim: #implements a logical OR for all outcomeTypes
                return True
        return False  

    def get_name(self):
        name = ""
        for outcome in self.outcomeTypes:
            name += OutcomeAssessor.DEATH if outcome == OutcomeAssessor.DEATH else str(outcome.value) + "-" 
        return name
