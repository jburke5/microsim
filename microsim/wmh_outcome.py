from microsim.outcome import Outcome, OutcomeType

class WMHOutcome(Outcome):
    
    phenotypeItems = ["sbi","wmh","wmhSeverityUnknown","wmhSeverity"]
    
    def __init__(self, fatal, sbi, wmh, wmhSeverityUnknown, wmhSeverity, priorToSim=False):
        self.fatal = fatal
        self.priorToSim = priorToSim
        super().__init__(OutcomeType.WMH, self.fatal, self.priorToSim)
        self.sbi = sbi
        self.wmh = wmh
        self.wmhSeverityUnknown = wmhSeverityUnknown
        self.wmhSeverity = wmhSeverity
        
    def __repr__(self):
        return f"""WMH Outcome: {self.type}, fatal: {self.fatal}, sbi: {self.sbi}, wmh: {self.wmh},
                   wmhSeverityUnknown: {self.wmhSeverityUnknown}, wmhSeverity: {self.wmhSeverity}"""
    
