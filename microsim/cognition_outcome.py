from microsim.outcome import Outcome, OutcomeType

class CognitionOutcome(Outcome):

    phenotypeItems = ["gcp"]

    def __init__(self, fatal, priorToSim, gcp):  
        self.fatal = fatal
        self.priorToSim = priorToSim
        super().__init__(OutcomeType.COGNITION, self.fatal, self.priorToSim)
        self.gcp = gcp
