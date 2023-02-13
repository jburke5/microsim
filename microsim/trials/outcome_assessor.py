from microsim.trials.attribute_outcome_assessor import AttributeOutcomeAssessor, AssessmentMethod

class OutcomeAssessor:
    DEATH = "death"
    CI = "ci" #cognitive impairment
    
    def __init__(self, outcomeTypes):
        self.outcomeTypes = outcomeTypes #assumed to be a python list

    def get_outcome(self, person): #RegressionAnalysis assumes that this method returns 0 or 1
        for outcomeType in self.outcomeTypes:
            if outcomeType == OutcomeAssessor.DEATH:
                outcomeDuringSim = person.is_dead()
            elif outcomeType == OutcomeAssessor.CI: #assesses if _gcp change was less than half SD of population GCP
                outcomeDuringSim = AttributeOutcomeAssessor(
                                                    "_gcp",
                                                    AssessmentMethod.CHANGELESSTHAN,
                                                    #SD was obtained from 300,000 NHANES population (not advanced) 
                                                    attributeParameter= -0.5*10.3099).get_outcome(person) 
            else:
                outcomeDuringSim = person.has_outcome_during_simulation(outcomeType)
            
            if outcomeDuringSim: #implements a logical OR for all outcomeTypes
                return True
        return False  

    def get_name(self):
        name = ""
        for outcome in self.outcomeTypes:
            name += outcome + "-" if (outcome==OutcomeAssessor.DEATH) | (outcome==OutcomeAssessor.CI) else str(outcome.value) + "-"
        return name
