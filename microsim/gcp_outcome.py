from microsim.outcome import Outcome, OutcomeType

class GCPOutcome(Outcome):

    phenotypeItems = ["gcp"]

    def __init__(self, fatal, selfReported, gcp):  
        self.fatal = fatal
        self.selfReported = selfReported
        super().__init__(OutcomeType.GLOBAL_COGNITIVE_PERFORMANCE, self.fatal, self.selfReported)
        self.gcp = gcp
