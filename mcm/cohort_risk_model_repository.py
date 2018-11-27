from mcm.regression_model import RegressionModel
from mcm.risk_model_repository import RiskModelRepository
from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

import json
import os


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
        abs_module_path = os.path.abspath(os.path.dirname(__file__))
        # TODO: need to get the "+" out of the path name
        model_spec_path = os.path.normpath(os.path.join(abs_module_path, "./data/",
                                                        modelName + "Spec.json"))
        with open(model_spec_path, 'r') as model_spec_file:
            model_spec = json.load(model_spec_file)
        model = RegressionModel(**model_spec)
        self._repository[referenceName] = StatsModelLinearRiskFactorModel(model, log)
