from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel
import numpy as np


class CoxRiskFactorModel(LinearRiskFactorModel):
    def __init__(self, regression_model, log_transform=False):
        super(CoxRiskFactorModel, self).__init__(regression_model, log_transform)
        self.one_year_linear_cumulative_hazard = (
            regression_model._one_year_linear_cumulative_hazard
        )
        self.one_year_quad_cumulative_hazard = regression_model._one_year_quad_cumulative_hazard

    def get_intercept(self):
        return 0

    def linear_predictor(self, person):
        return super(CoxRiskFactorModel, self).estimate_next_risk(person)

    def get_cumulative_hazard_for_interval(self, intervalStart, intervalEnd):
        cumHazardAtIntervalStart = (
            intervalStart * self.one_year_linear_cumulative_hazard
            + intervalStart**2 * self.one_year_quad_cumulative_hazard
        )
        cumHazardAtIntervalEnd = (
            intervalEnd * self.one_year_linear_cumulative_hazard
            + intervalEnd**2 * self.one_year_quad_cumulative_hazard
        )
        return cumHazardAtIntervalEnd - cumHazardAtIntervalStart

    def get_cumulative_hazard_for_years_in_sim(self, yearsInSim):
        return self.get_cumulative_hazard_for_interval(yearsInSim - 1, yearsInSim)

    def get_risk_for_person(self, person, years=1):
        if years != 1:
            raise NotImplementedError("CoxRiskFactorModel computes 1-year risk only")
        linear_predictor = self.linear_predictor(person)
        yearsInSim = len(person._age)
        return self.get_cumulative_hazard_for_years_in_sim(yearsInSim) * np.exp(float(linear_predictor))
