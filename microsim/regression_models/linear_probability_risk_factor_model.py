import numpy as np
from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel


class LinearProbabilityRiskFactorModel(LinearRiskFactorModel):
    def __init__(self, regression_model):
        super(LinearProbabilityRiskFactorModel, self).__init__(regression_model, False)

    def estimate_next_risk(self, person):
        linearRisk = super(LinearProbabilityRiskFactorModel, self).estimate_next_risk(
            person
        )
        riskWithResidual = linearRisk + self.draw_from_residual_distribution(person._rng)
        return riskWithResidual > 0.5
