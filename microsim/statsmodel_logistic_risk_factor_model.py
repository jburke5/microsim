from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

import numpy as np


class StatsModelLogisticRiskFactorModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model, log_transform=False):
        super().__init__(regression_model, log_transform)

    def estimate_linear_predictor(self, person):
        return super().estimate_next_risk(person)

    def logit(self, linearRisk):
        #return np.exp(linearRisk) / (1 + np.exp(linearRisk))
        if linearRisk<-10:
            risk = 0.
        elif linearRisk>10.:
            risk = 1.
        else:
            risk = 1/(1+np.exp(-linearRisk))
        return risk

    # apply inverse logit to the linear predictor
    def estimate_next_risk(self, person):
        return self.logit(self.estimate_linear_predictor(person))

