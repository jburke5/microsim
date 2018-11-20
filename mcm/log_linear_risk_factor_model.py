import numpy as np
from mcm.nhanes_linear_risk_factor_model import NHANESLinearRiskFactorModel


class LogLinearRiskFactorModel(NHANESLinearRiskFactorModel):
    def transform_linear_predictor(self, linear_pred):
        return np.exp(linear_pred)
