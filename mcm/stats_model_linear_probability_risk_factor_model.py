from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel


class StatsModelLinearProbabilityRiskFactorModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model):
        super(StatsModelLinearProbabilityRiskFactorModel, self).__init__(regression_model, False)

    # apply inverse logit to the linear predictor
    def estimate_next_risk(self, person):
        linearRisk = super(StatsModelLinearProbabilityRiskFactorModel, self).estimate_next_risk(person)
        return linearRisk > 0.5
