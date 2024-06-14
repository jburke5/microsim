from microsim.outcome import Outcome, OutcomeType

class QALYOutcome(Outcome):

    phenotypeItems = ["qaly"]

    def __init__(self, fatal, selfReported, qaly):  
        self.fatal = fatal
        self.selfReported = selfReported
        super().__init__(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, self.fatal, self.selfReported)
        self.qaly = qaly
