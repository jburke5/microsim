from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel


class StatsModelRoundedLinearRiskFactorModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model):
        super(StatsModelRoundedLinearRiskFactorModel, self).__init__(regression_model, False)

    # apply inverse logit to the linear predictor
    def estimate_next_risk(self, person):
        linearRisk = round(super(StatsModelRoundedLinearRiskFactorModel, self).estimate_next_risk(person))
        return linearRisk if linearRisk > 0 else 0
