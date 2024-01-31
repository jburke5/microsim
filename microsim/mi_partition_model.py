from microsim.outcome import OutcomeType, Outcome

class MIPartitionModel:
    """Fatal mi probability from: Wadhera, R. K., Joynt Maddox, K. E., Wang, Y., Shen, C., Bhatt, D. L., & Yeh, R. W. (2018). 
       Association Between 30-Day Episode Payments and Acute Myocardial Infarction Outcomes Among Medicare Beneficiaries. 
       Circ. Cardiovasc. Qual. Outcomes, 11(3), e46–9. http://doi.org/10.1161/CIRCOUTCOMES.117.004397"""

    def __init__(self):
        self._mi_case_fatality = 0.13
        self._mi_secondary_case_fatality = 0.13
        
    def update_cv_outcome(self, person, fatal):
        #need to double check this
        person._outcomes[OutcomeType.CARDIOVASCULAR][-1].fatal = fatal    
    
    def will_have_fatal_mi(self, person):
        fatalMIProb = self._mi_case_fatality
        fatalProb = self._mi_secondary_case_fatality if person._mi else fatalMIProb
        return person._rng.uniform(size=1) < fatalProb
    
    def generate_next_outcome(self, person):
        fatal = self.will_have_fatal_mi(person)
        return Outcome(outcomeType.MI, fatal)
        
    def get_next_outcome(self, person):
        if person.has_outcome_at_current_age(OutcomeType.CARDIOVASCULAR):
            #assumes that stroke has been decided by now
            if person.has_outcome_at_current_age(OutcomeType.STROKE):
                return None
            else:
                miOutcome = self.generate_next_outcome(person)
                self.update_cv_outcome(person, miOutcome.fatal)
                return miOutcome
