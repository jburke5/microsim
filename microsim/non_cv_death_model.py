from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel
from microsim.data_loader import load_model_spec
from microsim.regression_model import RegressionModel
from microsim.outcome import Outcome, OutcomeType
from microsim.modality import Modality

import numpy as np

#this class represents non CV mortality
class NonCVDeathModel(StatsModelLogisticRiskFactorModel):
    def __init__(self, wmhSpecific=True):
        modelSpec = load_model_spec("nhanesMortalityModelLogit")
        # Recalibrate mortalitly model to align with life table data, as explored in notebook buildNHANESMortalityModel
        modelSpec["coefficients"]["age"] = modelSpec["coefficients"]["age"]*(-1)
        modelSpec["coefficients"]["squareAge"] = modelSpec["coefficients"]["squareAge"]*4 
        super().__init__(RegressionModel(**modelSpec), False)
        self.wmhSpecific=wmhSpecific
        
    def generate_next_outcome(self, person):
        fatal=True
        selfReported = False
        return Outcome(OutcomeType.NONCARDIOVASCULAR, fatal, selfReported)
        
    def get_next_outcome(self, person):
        #need to find a better way to check for mortality during current age outcomes
        if person.has_fatal_outcome_at_current_age(OutcomeType.CARDIOVASCULAR):
            return None
        else:
            if person._rng.uniform(size=1)<self.estimate_next_risk(person):
                return self.generate_next_outcome(person) 
            else:
                return None
        
    def get_scd_term(self, person):
        '''This term, for silent cerebrovascular disease, is based on time-dependent hazard ratios (see Clancy2024 paper).
        The linear models for the time-dependent hazard ratios were obtain by a LLS fit to the values shown in the Clancy2024 paper.
        In addition, because the non cv death model is a logistic model, we scale the hazard ratios.'''
        if not person._modality == Modality.NO.value: #if there was a brain scan      
            if self.wmhSpecific:
                scdTerm = 0.35 #this modifies the intercept
                scalingMriSbi = 1.25 #these are the four scaling factors so that I can use the hazard ratios in the logistic non cv death  model
                scalingMriWmh = 0.16667
                scalingCtSbi = 0.64
                scalingCtWmh = 0.01
                if person._outcomes[OutcomeType.WMH][0][1].sbi:
                    scdTerm += np.log(1.76 - 0.007513 * person._age[-1])
                    if person._modality == Modality.CT.value:
                        scdTerm = scalingCtSbi * scdTerm
                    elif person._modality == Modality.MR.value:
                        scdTerm = scalingMriSbi * scdTerm
                    else:
                        raise RuntimeError("Person has SBI but no modality")
                if person._outcomes[OutcomeType.WMH][0][1].wmh:
                    if person._modality == Modality.MR.value:
                        scdTerm += scalingMriWmh * np.log(1.645 - 0.0068653 * person._age[-1])
                    elif person._modality == Modality.CT.value:
                        scdTerm += scalingCtWmh * np.log(1.865 - 0.0081088 * person._age[-1])
                    else:
                        raise RuntimeError("Person has WMH but no modality")
            else:
                scdTerm = 0.35 #represents average risk of the kaiser population, obtained through a different optimization than the one above  
        else:
            scdTerm = 0. 
        return scdTerm
        
    def estimate_linear_predictor(self, person):
        lp = super().estimate_linear_predictor(person) + self.get_scd_term(person) #nhanes model + adjustment for kaiser population
        return lp
