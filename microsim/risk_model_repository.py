from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.stats_model_linear_probability_risk_factor_model import StatsModelLinearProbabilityRiskFactorModel
from microsim.stats_model_rounded_linear_risk_factor_model import StatsModelRoundedLinearRiskFactorModel
from microsim.data_loader import load_regression_model
from microsim.risk_factor import DynamicRiskFactorsType

class RiskModelRepository:
    def __init__(self):
        self._repository = {}
        #bounds based on NHANES data from 1999 to 20017 (all data), 0.9*nhanesMin, 1.1*nhanesMax
        #age is an exception, bounds set manually
        self._lowerBounds = {DynamicRiskFactorsType.SBP.value: 58.20,
                             DynamicRiskFactorsType.DBP.value: 36. ,
                             DynamicRiskFactorsType.CREATININE.value: 0.090,
                             DynamicRiskFactorsType.WAIST.value: 49.95,
                             DynamicRiskFactorsType.LDL.value: 8.10,
                             DynamicRiskFactorsType.A1C.value: 1.80,
                             DynamicRiskFactorsType.TRIG.value: 9.,
                             DynamicRiskFactorsType.BMI.value: 10.836,
                             DynamicRiskFactorsType.HDL.value: 5.4,
                             DynamicRiskFactorsType.AGE.value: 18,
                             DynamicRiskFactorsType.TOT_CHOL.value: 53.1}
        self._upperBounds = {DynamicRiskFactorsType.SBP.value: 297.,
                             DynamicRiskFactorsType.DBP.value: 152.53,
                             DynamicRiskFactorsType.CREATININE.value: 19.58,
                             DynamicRiskFactorsType.WAIST.value: 196.9,
                             DynamicRiskFactorsType.LDL.value: 691.9,
                             DynamicRiskFactorsType.A1C.value: 20.68,
                             DynamicRiskFactorsType.TRIG.value: 4656.3,
                             DynamicRiskFactorsType.BMI.value: 143.23,
                             DynamicRiskFactorsType.HDL.value: 248.6,
                             DynamicRiskFactorsType.AGE.value: 130,
                             DynamicRiskFactorsType.TOT_CHOL.value: 894.3}

    def apply_bounds(self, varName, varValue):
        """
        Ensures that risk factor are within static prespecified bounds.

        Other algorithms might be needed in the future to avoid pooling in the tails,
        if there are many extreme risk factor results.
        """
        if varName in self._upperBounds:
            upperBound = self._upperBounds[varName]
            varValue = varValue if varValue < upperBound else upperBound
        if varName in self._lowerBounds:
            lowerBound = self._lowerBounds[varName]
            varValue = varValue if varValue > lowerBound else lowerBound
        return varValue

    def get_model(self, name):
        return self._repository[name]

    def _initialize_linear_risk_model(self, referenceName, modelName, log=False):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelLinearRiskFactorModel(model, log)

    def _initialize_linear_probability_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelLinearProbabilityRiskFactorModel(model)

    def _initialize_int_rounded_linear_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelRoundedLinearRiskFactorModel(model)
