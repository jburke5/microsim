from mcm.risk_model_repository import RiskModelRepository
from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

from statsmodels.regression.linear_model import OLSResults


class CohortRiskModelRepository(RiskModelRepository):
    def __init__(self):
        super(CohortRiskModelRepository, self).__init__()
        self._initialize_linear_risk_model("hdl", "hdlCohortModel")
        self._initialize_linear_risk_model("bmi", "bmiCohortModel")
        self._initialize_linear_risk_model("totChol", "totCholCohortModel")
        self._initialize_linear_risk_model("trig", "trigCohortModel")
        self._initialize_linear_risk_model("a1c", "a1cCohortModel")
        self._initialize_linear_risk_model("ldl", "ldlCohortModel")
        self._initialize_linear_risk_model("sbp", "logSBPCohortModel", log=True)
        self._initialize_linear_risk_model("dbp", "logDBPCohortModel", log=True)

    def _initialize_linear_risk_model(self, referenceName, modelName, log=False):
        model = OLSResults.load("mcm/data/" + modelName + ".pickle")
        self._repository[referenceName] = StatsModelLinearRiskFactorModel(model, log)
