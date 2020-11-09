from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
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

    # need to override for specific subclasses that implement it.
    def linear_predictor_vectorized(self, person):
        return None

    def get_cumulative_hazard_for_interval(self, intervalStart, intervalEnd):
        cumHazardAtIntervalStart = intervalStart * self.one_year_linear_cumulative_hazard + \
            intervalStart**2 * self.one_year_quad_cumulative_hazard
        cumHazardAtIntervalEnd = intervalEnd * self.one_year_linear_cumulative_hazard + \
            intervalEnd**2 * self.one_year_quad_cumulative_hazard
        return cumHazardAtIntervalEnd - cumHazardAtIntervalStart

    def get_cumulative_hazard_for_years_in_sim(self, yearsInSim):
        return self.get_cumulative_hazard_for_interval(yearsInSim - 1, yearsInSim)

    def get_risk_for_person(self, person, years, vectorized=False):
        linear_predictor = self.linear_predictor_vectorized(person) if vectorized else self.linear_predictor(person)
        yearsInSim = person.totalYearsInSim if vectorized else len(person._age)
        return self.get_cumulative_hazard_for_years_in_sim(yearsInSim) * np.exp(linear_predictor)
