from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
import numpy as np


class StatsModelCoxModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model, log_transform=False):
        super(StatsModelCoxModel, self).__init__(regression_model, log_transform)
        self.one_year_linear_cumulative_hazard = \
            regression_model._one_year_linear_cumulative_hazard
        self.one_year_quad_cumulative_hazard = regression_model._one_year_quad_cumulative_hazard

    def get_intercept(self):
        return 0

    def linear_predictor(self, person):
        return super(StatsModelCoxModel, self).estimate_next_risk(person)

    def get_cumulative_hazard_for_interval(self, intervalStart, intervalEnd):
        cumHazardAtIntervalStart = intervalStart * self.one_year_linear_cumulative_hazard + \
            intervalStart**2 * self.one_year_quad_cumulative_hazard
        cumHazardOneIntervalEnd = intervalEnd * self.one_year_linear_cumulative_hazard + \
            intervalEnd**2 * self.one_year_quad_cumulative_hazard
        return cumHazardOneIntervalEnd - cumHazardAtIntervalStart

    def get_cumulative_hazard(self, person):
        return self.get_cumulative_hazard_for_interval(len(person._age) - 1, len(person._age))

    def get_risk_for_person(self, person, years):
        return self.get_cumulative_hazard(person) * np.exp(self.linear_predictor(person))
