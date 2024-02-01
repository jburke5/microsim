from microsim.outcome import Outcome, OutcomeType

class QALYOutcome(Outcome):

    phenotypeItems = ["qaly"]

    def __init__(self, fatal, qaly):  
        self.fatal = fatal
        super().__init__(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, self.fatal)
        self.qaly = qaly
