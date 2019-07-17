from mcm.risk_model_repository import RiskModelRepository
from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from mcm.stats_model_linear_probability_risk_factor_model import StatsModelLinearProbabilityRiskFactorModel
from mcm.stats_model_rounded_linear_risk_factor_model import StatsModelRoundedLinearRiskFactorModel
from mcm.data_loader import load_regression_model

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
        self._initialize_linear_probability_risk_model("anyPhysicalActivity", "anyPhysicalActivityCohortModel")
        self._initialize_linear_probability_risk_model("afib", "afibCohortModel")
        self._initialize_linear_probability_risk_model("statin", "statinCohortModel")
        self._initialize_int_rounded_linear_risk_model("antiHypertensiveCount", "antiHypertensiveCountCohortModel")
        self._initialize_linear_risk_model("sbp", "logSbpCohortModel", log=True)
        self._initialize_linear_risk_model("dbp", "logDbpCohortModel", log=True)

    def _initialize_linear_risk_model(self, referenceName, modelName, log=False):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelLinearRiskFactorModel(model, log)

    def _initialize_linear_probability_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelLinearProbabilityRiskFactorModel(model)

    def _initialize_int_rounded_linear_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelRoundedLinearRiskFactorModel(model)
