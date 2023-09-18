import numpy as np
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

#this class was designed to be used in implementations of multinomial logistic regression models
#note: odds and relative risks are not the same thing, see Stata's mlogit technical note
class StatsModelRelRiskFactorModel(StatsModelLinearRiskFactorModel):
    
    def __init__(self, regressionModel):
        self._regressionModel = regressionModel
        super().__init__(self._regressionModel)

    def estimate_rel_risk(self, person):
        return np.exp(super().estimate_next_risk(person))
    
    def estimate_rel_risk_vectorized(self, person):
        return np.exp(super().estimate_next_risk_vectorized(person))
