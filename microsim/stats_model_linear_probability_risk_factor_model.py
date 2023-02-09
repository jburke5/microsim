import numpy as np
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel


class StatsModelLinearProbabilityRiskFactorModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model):
        super(StatsModelLinearProbabilityRiskFactorModel, self).__init__(regression_model, False)

    def estimate_next_risk(self, person, rng=None):
        #rng = np.random.default_rng(rng)
        linearRisk = super(StatsModelLinearProbabilityRiskFactorModel, self).estimate_next_risk(
            person
        )
        riskWithResidual = linearRisk + self.draw_from_residual_distribution(rng)
        return riskWithResidual > 0.5

    def estimate_next_risk_vectorized(self, x, rng=None):
        #rng = np.random.default_rng(rng)
        linearRisk = super(
            StatsModelLinearProbabilityRiskFactorModel, self
        ).estimate_next_risk_vectorized(x)
        riskWithResidual = linearRisk + self.draw_from_residual_distribution(rng)
        return riskWithResidual > 0.5
