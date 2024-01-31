from microsim.data_loader import load_regression_model
from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

#the intercept of this model was modified in order to have agreement with the 2019 global burden of disease data
#optimization of the intercept was performed on the afibModelRecalibrations notebook
class AFibPrevalenceModel(StatsModelLogisticRiskFactorModel):
    def __init__(self):
        super().__init__(load_regression_model("BaselineAFibModel"))
    
    def estimate_next_risk(self, person):
        return person._rng.uniform() < super().estimate_next_risk(person)

#moved away from linear probability risk factor model because this approach gives the least absolute deviations in afib versus the
#global burden of disease data
#the intercept and age coefficient of the cohort afib model were modified to fit the gbd data
#note that the riskWithResidual is not bounded by 0, 1 but the rng.uniform is 
class AFibIncidenceModel(StatsModelLinearRiskFactorModel):
    def __init__(self):
        regression_model = load_regression_model("afibCohortModel")
        super().__init__(regression_model, False)

    def estimate_next_risk(self, person, rng=None):
        linearRisk = super().estimate_next_risk(person)
        riskWithResidual = linearRisk + self.draw_from_residual_distribution(rng)
        return person._rng.uniform() < riskWithResidual

    def estimate_next_risk_vectorized(self, x, rng=None):
        linearRisk = super().estimate_next_risk_vectorized(x)
        riskWithResidual = linearRisk + self.draw_from_residual_distribution(rng)
        return rng.uniform()  < riskWithResidual
