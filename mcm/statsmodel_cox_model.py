from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
import numpy as np


class StatsModelCoxModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model, log_transform=False):
        super(StatsModelCoxModel, self).__init__(regression_model, log_transform)

    # called by superclass...
    def initialize_model_params(self, regression_model, log_transform):
        self.parameters = regression_model._coefficients
        self.standard_errors = regression_model._coefficient_standard_errors
        self.one_year_cumulative_hazard = regression_model._one_year_cumulative_hazard
        self.log_transform = log_transform

    def get_intercept(self):
        return 0

    def estimate_next_risk(self, person):
        linear_predictor = super(StatsModelCoxModel, self).estimate_next_risk(person)
        return self.one_year_cumulative_hazard * np.exp(linear_predictor)
