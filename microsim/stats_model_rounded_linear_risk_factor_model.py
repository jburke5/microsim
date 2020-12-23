from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel


class StatsModelRoundedLinearRiskFactorModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model):
        super(StatsModelRoundedLinearRiskFactorModel, self).__init__(regression_model, False)

    # apply inverse logit to the linear predictor
    def estimate_next_risk(self, person):
        linearRisk = super(StatsModelRoundedLinearRiskFactorModel, self).estimate_next_risk(person)
        riskWithResidual = round(linearRisk + self.draw_from_residual_distribution())
        return riskWithResidual if riskWithResidual > 0 else 0

    def estimate_next_risk_vectorized(self, x):
        linearRisk = super(StatsModelRoundedLinearRiskFactorModel,
                           self).estimate_next_risk_vectorized(x)
        riskWithResidual = round(linearRisk + self.draw_from_residual_distribution())
        return riskWithResidual if riskWithResidual > 0 else 0
