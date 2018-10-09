import numpy as np
from mcm.linear_risk_factor_model import LinearRiskFactorModel


class LogLinearRiskFactorModel(LinearRiskFactorModel):
    def transformLinearPredictor(self, linear_pred):
        return np.exp(linear_pred)
