from microsim.sbi_model import SBIModel
from microsim.wmh_presence_model import WMHPresenceModel
from microsim.wmh_outcome import WMHOutcome
from microsim.wmh_severity_unknown import WMHSeverityUnknownModel, WMHSeverityUnknown
from microsim.wmh_severity import WMHSeverityModel, WMHSeverity
from microsim.outcome import OutcomeType

class WMHModel:
    """White Matter Hypodensity model."""
    def __init__(self):
        self.sbiModel = SBIModel()
        self.wmhPresenceModel = WMHPresenceModel()
        self.wmhSeverityUnknownModel = WMHSeverityUnknownModel()
        self.wmhSeverityModel = WMHSeverityModel()
    
    def generate_next_outcome(self, person):
        fatal = False
        sbi = self.sbiModel.estimate_next_risk(person)
        wmh = self.wmhPresenceModel.estimate_next_risk(person)
        if wmh:
            wmhSeverityUnknown = self.wmhSeverityUnknownModel.estimate_next_risk(person)
            wmhSeverity = self.wmhSeverityModel.estimate_next_risk(person)
        else:
            wmhSeverityUnknown = WMHSeverityUnknown.UNKNOWN
            wmhSeverity = WMHSeverity.NO
        return WMHOutcome(fatal, sbi, wmh, wmhSeverityUnknown, wmhSeverity, priorToSim = False)
        
    def get_next_outcome(self, person):
        if len(person._outcomes[OutcomeType.WMH])==0:
            return self.generate_next_outcome(person)
        else:
            return None
