from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.stats_model_linear_probability_risk_factor_model import StatsModelLinearProbabilityRiskFactorModel
from microsim.stats_model_rounded_linear_risk_factor_model import StatsModelRoundedLinearRiskFactorModel
from microsim.data_loader import load_regression_model

class RiskModelRepository:
    def __init__(self):
        self._repository = {}

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
