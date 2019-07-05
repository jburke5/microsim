from mcm.regression_model import RegressionModel
from mcm.risk_model_repository import RiskModelRepository
from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from mcm.stats_model_linear_probability_risk_factor_model import StatsModelLinearProbabilityRiskFactorModel
from mcm.stats_model_rounded_linear_risk_factor_model import StatsModelRoundedLinearRiskFactorModel

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
        self._initialize_linear_risk_model("waist", "waistCohortModel")
        # left off here...
        self._initialize_linear_probability_risk_model("anyPhysicalActivity", "anyPhysicalActivityCohortModel")
        self._initialize_linear_probability_risk_model("afib", "afibCohortModel")
        self._initialize_linear_probability_risk_model("statin", "statinCohortModel")
        self._initialize_int_rounded_linear_risk_model("antiHypertensiveCount", "antiHypertensiveCountCohortModel")
        self._initialize_linear_risk_model("sbp", "logSbpCohortModel", log=True)
        self._initialize_linear_risk_model("dbp", "logDbpCohortModel", log=True)

    def _load_model_spec(self, referenceName, modelName):
        abs_module_path = os.path.abspath(os.path.dirname(__file__))
        model_spec_path = os.path.normpath(os.path.join(abs_module_path, "./data/",
                                                        modelName + "Spec.json"))
        with open(model_spec_path, 'r') as model_spec_file:
            model_spec = json.load(model_spec_file)
        return RegressionModel(**model_spec)

    def _initialize_linear_risk_model(self, referenceName, modelName, log=False):
        model = self._load_model_spec(referenceName, modelName)
        self._repository[referenceName] = StatsModelLinearRiskFactorModel(model, log)

    def _initialize_linear_probability_risk_model(self, referenceName, modelName):
        model = self._load_model_spec(referenceName, modelName)
        self._repository[referenceName] = StatsModelLinearProbabilityRiskFactorModel(model)

    def _initialize_int_rounded_linear_risk_model(self, referenceName, modelName):
        model = self._load_model_spec(referenceName, modelName)
        self._repository[referenceName] = StatsModelRoundedLinearRiskFactorModel(model)

