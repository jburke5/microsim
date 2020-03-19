from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

import numpy as np


class StatsModelLogisticRiskFactorModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model, log_transform=False):
        super(StatsModelLogisticRiskFactorModel, self).__init__(regression_model, log_transform)

    # apply inverse logit to the linear predictor
    def estimate_next_risk(self, person):
        linearRisk = super(StatsModelLogisticRiskFactorModel, self).estimate_next_risk(person)
        return np.random.rand() < np.exp(linearRisk) / (1 + np.exp(linearRisk))
