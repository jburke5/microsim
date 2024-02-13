from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.stats_model_linear_probability_risk_factor_model import StatsModelLinearProbabilityRiskFactorModel
from microsim.stats_model_rounded_linear_risk_factor_model import StatsModelRoundedLinearRiskFactorModel
from microsim.data_loader import load_regression_model
from microsim.risk_factor import DynamicRiskFactorsType

class RiskModelRepository:
    def __init__(self):
        self._repository = {}
        self._lowerBounds = {DynamicRiskFactorsType.SBP.value: 60,
                             DynamicRiskFactorsType.DBP.value: 20,
                             DynamicRiskFactorsType.CREATININE.value: 0.1}
        self._upperBounds = {DynamicRiskFactorsType.SBP.value: 300,
                             DynamicRiskFactorsType.DBP.value: 180}

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
