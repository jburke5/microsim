from microsim.sbi_model import SBIModel
from microsim.wmh_outcome import WMHOutcome
from microsim.wmh_severity_unknown import WMHSeverityUnknownModel
from microsim.wmh_severity import WMHSeverityCTModel, WMHSeverityMRModel, WMHSeverity
from microsim.outcome import OutcomeType
from microsim.modality import Modality

class WMHModel:
    """White Matter Hypodensity model."""
    def __init__(self):
        self.sbiModel = SBIModel()
        self.wmhSeverityUnknownModel = WMHSeverityUnknownModel()
        self.wmhSeverityCTModel = WMHSeverityCTModel()
        self.wmhSeverityMRModel = WMHSeverityMRModel()
    
    def generate_next_outcome(self, person):
        '''The severity unknown model was derived based on the entire population, which is why this is used first.
        When the severity was unknown, WMH was present in the population but without the severity known, which is why
        wmh is set to True when the severity was unknown.
        We use the ordinal logistic model WMHSeverity because that is the model that was derived by excluding the
        severity unknown part of the population.
        Also, the WMHSeverity.NO severity predicts the part of the population that did not have WMH.
        I will not use the WMHPresence model because that was derived by including the WMHSeverity unknown part of the population.
        For more details, see email on 7/12/2024.'''
        fatal = False
        sbi = self.sbiModel.estimate_next_risk(person)
        wmhSeverityUnknown = self.wmhSeverityUnknownModel.estimate_next_risk(person)
        if wmhSeverityUnknown==False:
            if person._modality == Modality.CT.value:
                wmhSeverity = self.wmhSeverityCTModel.estimate_next_risk(person)
                wmh = False if wmhSeverity == WMHSeverity.NO else True
            elif person._modality == Modality.MR.value:
                wmhSeverity = self.wmhSeverityMRModel.estimate_next_risk(person)
                wmh = False if wmhSeverity == WMHSeverity.NO else True
        else:
            wmhSeverity = None
            wmh = True
        return WMHOutcome(fatal, sbi, wmh, wmhSeverityUnknown, wmhSeverity, priorToSim = False)
        
    def get_next_outcome(self, person):
        if len(person._outcomes[OutcomeType.WMH])==0:
            return self.generate_next_outcome(person)
        else:
            return None
