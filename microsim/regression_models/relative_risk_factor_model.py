import numpy as np
from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel

#this class was designed to be used in implementations of multinomial logistic regression models
#note: odds and relative risks are not the same thing, see Stata's mlogit technical note
class RelativeRiskFactorModel(LinearRiskFactorModel):
    
    def __init__(self, regressionModel):
        self._regressionModel = regressionModel
        super().__init__(self._regressionModel)

    def estimate_rel_risk(self, person):
        return np.exp(super().estimate_next_risk(person))
