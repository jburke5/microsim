from microsim.risk_factors.risk_model_repository import RiskModelRepository
from microsim.risk_factors.nhanes_linear_risk_factor_model import NHANESLinearRiskFactorModel
from microsim.test.static_risk_factor_over_time_repository import DoNothingModel

import pandas as pd
import numpy as np


class TestRiskModelRepository(RiskModelRepository):
    params = {
        "age": 1.0,
        "gender": 0,
        "raceEthnicity[T.2]": 0,
        "raceEthnicity[T.3]": 0,
        "raceEthnicity[T.4]": 0,
        "raceEthnicity[T.5]": 0,
        "sbp": 0.5,
        "dbp": 0,
        "a1c": 0,
        "hdl": 0,
        "bmi": 0,
        "smokingStatus[T.1]": 0,
        "smokingStatus[T.2]": 0,
        "totChol": 0,
        "Intercept": 80,
    }

    def __init__(self, nullModels = False):
        super(TestRiskModelRepository, self).__init__()

        for name in ["sbp", "dbp", "ldl", "trig", "a1c", "hdl", "totChol",
                    "bmi", "waist", "anyPhysicalActivity", "statin", "antiHypertensiveCount",
                    "afib", "alcoholPerWeek", "creatinine"]:
            if nullModels:
                self.set_null_model_for_name(name)
            else:
                self.set_default_model_for_name(name)
        for name in ["age", "pvd"]:
            self.set_null_model_for_name(name) 


    def set_default_model_for_name(self, name):
        self._repository[name] = NHANESLinearRiskFactorModel(params=TestRiskModelRepository.params, resids = pd.Series(np.zeros(10)))


    def set_null_model_for_name(self, name):
        self._repository[name] = DoNothingModel(name)
