from microsim.regression_models.logistic_risk_factor_model import LogisticRiskFactorModel

# from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel

import numpy as np


class RandInterceptLogisticRiskFactorModel(LogisticRiskFactorModel):
    def __init__(self, regression_model, log_transform=False, rand_intercept_name=None):
        super().__init__(regression_model, log_transform)
        self._rand_intercept_name = rand_intercept_name
        self._rand_intercept_sd = regression_model._residual_standard_deviation
        self._rand_intercept_mean = regression_model._residual_mean

    # apply inverse logit to the linear predictor and add the random intercept
    def estimate_next_risk(self, person):
        linearRisk = super().estimate_linear_predictor(person)
        rand_intercept = person._randomEffects[self._rand_intercept_name]
        totalRisk = linearRisk + rand_intercept
        return np.exp(totalRisk) / (1 + np.exp(totalRisk))
