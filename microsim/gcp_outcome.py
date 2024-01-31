from microsim.outcome import Outcome, OutcomeType

class GCPOutcome(Outcome):

    phenotypeItems = ["gcp"]

    def __init__(self, fatal, gcp):  
        self.fatal = fatal
        super().__init__(OutcomeType.GLOBAL_COGNITIVE_PERFORMANCE, self.fatal)
        self.gcp = gcp
