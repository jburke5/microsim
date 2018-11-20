from mcm.nhanes_linear_risk_factor_model import NHANESLinearRiskFactorModel
from mcm.log_linear_risk_factor_model import LogLinearRiskFactorModel
from mcm.risk_model_repository import RiskModelRepository

from statsmodels.regression.linear_model import OLSResults


class NHANESRiskModelRepository(RiskModelRepository):
    def __init__(self):
        super(NHANESRiskModelRepository, self).__init__()
        self._initialize_linear_risk_model("hdl", "matchedHdlModel")
        self._initialize_linear_risk_model("bmi", "matchedBmiModel")
        self._initialize_linear_risk_model("totChol", "matchedTotCholModel")
        self._initialize_linear_risk_model("a1c", "matchedA1cModel")
        self._initialize_linear_risk_model("bmi", "matchedBmiModel")
        self._initialize_log_linear_risk_model("sbp", "logSBPModel")
        self._initialize_log_linear_risk_model("dbp", "logDBPModel")

    def _initialize_linear_risk_model(self, referenceName, modelName):
        modelResults = OLSResults.load("mcm/data/" + modelName + ".pickle")
        self._repository[referenceName] = NHANESLinearRiskFactorModel(
            referenceName, modelResults.params, modelResults.bse, modelResults.resid)

    def _initialize_log_linear_risk_model(self, referenceName, modelName):
        modelResults = OLSResults.load("mcm/data/" + modelName + ".pickle")
        self._repository[referenceName] = LogLinearRiskFactorModel(
            referenceName, modelResults.params, modelResults.bse, modelResults.resid)
