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

    def get_outcome_time(self, person):
        # initiatilize with the last follow-up time, which will be either when the simulation 
        # stops or the time of death
        times = [len(person._age)-1]
        #print(self.outcomeTypes)
        outcomeTypesForTime = self.outcomeTypes.copy()
        if OutcomeAssessor.DEATH in outcomeTypesForTime:
            outcomeTypesForTime.remove(OutcomeAssessor.DEATH)
        for outcomeType in outcomeTypesForTime:
            if outcomeType == OutcomeAssessor.CI: #assesses if _gcp change was less than half SD of population GCP
                times.append(AttributeOutcomeAssessor(
                                                    "_gcp",
                                                    AssessmentMethod.INCREMENTALCHANGELESSTHAN,
                                                    #SD was obtained from 300,000 NHANES population (not advanced) 
                                                    attributeParameter= -0.5*10.3099).get_outcome_time(person)) 
            else:
                if person.has_outcome_during_simulation(outcomeType):
                    ageAtFirstOutcome = person.get_age_at_first_outcome_in_sim(outcomeType)  
                    times.append(ageAtFirstOutcome - person._age[0]-1)            
        return min(times)


    def get_name(self):
        name = ""
        for outcome in self.outcomeTypes:
            name += outcome + "-" if (outcome==OutcomeAssessor.DEATH) | (outcome==OutcomeAssessor.CI) else str(outcome.value) + "-"
        return name
