from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel
from microsim.data_loader import load_model_spec
from microsim.regression_model import RegressionModel
from microsim.outcome import Outcome, OutcomeType

#this class represents non CV mortality
class NonCVDeathModel(StatsModelLogisticRiskFactorModel):
    def __init__(self):
        modelSpec = load_model_spec("nhanesMortalityModelLogit")
        # Recalibrate mortalitly model to align with life table data, as explored in notebook buildNHANESMortalityModel
        modelSpec["coefficients"]["age"] = modelSpec["coefficients"]["age"]*(-1)
        modelSpec["coefficients"]["squareAge"] = modelSpec["coefficients"]["squareAge"]*4 
        super().__init__(RegressionModel(**modelSpec), False)
        
    def generate_next_outcome(self, person):
        fatal=True
        selfReported = False
        return Outcome(OutcomeType.NONCARDIOVASCULAR, fatal, selfReported)
        
    def get_next_outcome(self, person):
        #need to find a better way to check for mortality during current age outcomes
        if person.has_fatal_outcome_at_current_age(OutcomeType.CARDIOVASCULAR):
            return None
        else:
            #if person._rng.uniform(size=1)<self.get_risk_for_person(person, person._rng, 1):
            if person._rng.uniform(size=1)<self.estimate_next_risk(person):
                return self.generate_next_outcome(person) 
            else:
                return None
        
