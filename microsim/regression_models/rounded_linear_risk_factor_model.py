import numpy as np
from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel


class RoundedLinearRiskFactorModel(LinearRiskFactorModel):
    def __init__(self, regression_model):
        super(RoundedLinearRiskFactorModel, self).__init__(regression_model, False)

    # apply inverse logit to the linear predictor
    def estimate_next_risk(self, person):
        linearRisk = super(RoundedLinearRiskFactorModel, self).estimate_next_risk(person)
        riskWithResidual = round(linearRisk + self.draw_from_residual_distribution(person._rng))
        return riskWithResidual if riskWithResidual > 0 else 0
