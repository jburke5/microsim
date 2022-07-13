from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel

# from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

import numpy as np


class StatsModelRandInterceptLogisticRiskFactorModel(StatsModelLogisticRiskFactorModel):
    def __init__(self, regression_model, log_transform=False, rand_intercept_name=None):
        super().__init__(regression_model, log_transform)
        self._rand_intercept_name = rand_intercept_name
        self._rand_intercept_sd = regression_model._residual_standard_deviation
        self._rand_intercept_mean = regression_model._residual_mean

    def get_random_intercept_name_vectorized(self):
        return self._rand_intercept_name + "RandomEffect"

    # apply inverse logit to the linear predictor and add the random intercept
    def estimate_next_risk(self, person):
        linearRisk = super().estimate_linear_predictor(person)
        rand_intercept = person._randomEffects[self._rand_intercept_name]
        totalRisk = linearRisk + rand_intercept
        return np.exp(totalRisk) / (1 + np.exp(totalRisk))

    def estimate_next_risk_vectorized(self, x):
        linearRisk = super().estimate_linear_predictor_vectorized(x)
        rand_intercept = x[self.get_random_intercept_name_vectorized()]
        totalRisk = linearRisk + rand_intercept

        return np.exp(totalRisk) / (1 + np.exp(totalRisk))
