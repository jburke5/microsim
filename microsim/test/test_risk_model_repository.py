from microsim.risk_model_repository import RiskModelRepository
from microsim.nhanes_linear_risk_factor_model import NHANESLinearRiskFactorModel

import pandas as pd
import numpy as np


class TestRiskModelRepository(RiskModelRepository):
    def __init__(self):
        super(TestRiskModelRepository, self).__init__()

        params = {
            'age': 1.0,
            'gender': 0,
            'raceEthnicity[T.2]': 0,
            'raceEthnicity[T.3]': 0,
            'raceEthnicity[T.4]': 0,
            'raceEthnicity[T.5]': 0,
            'sbp': 0.5,
            'dbp': 0,
            'a1c': 0,
            'hdl': 0,
            'bmi': 0,
            'smokingStatus[T.1]': 0,
            'smokingStatus[T.2]': 0,
            'totChol': 0,
            'Intercept': 80,
        }

        ses = {
            'age': 0,
            'gender': 0,
            'raceEthnicity[T.2]': 0,
            'raceEthnicity[T.3]': 0,
            'raceEthnicity[T.4]': 0,
            'raceEthnicity[T.5]': 0,
            'smokingStatus[T.1]': 0,
            'smokingStatus[T.2]': 0,
            'sbp': 0,
            'dbp': 0,
            'a1c': 0,
            'hdl': 0,
            'bmi': 0,
            'totChol': 0,
            'Intercept': 0,
        }

        self._repository['sbp'] = NHANESLinearRiskFactorModel('sbp', params=params, ses=ses,
                                                              resids=pd.Series(np.zeros(10)))
        self._repository['dbp'] = NHANESLinearRiskFactorModel('dbp', params=params, ses=ses,
                                                              resids=pd.Series(np.zeros(10)))
        self._repository['ldl'] = NHANESLinearRiskFactorModel('ldl', params=params, ses=ses,
                                                              resids=pd.Series(np.zeros(10)))
        self._repository['trig'] = NHANESLinearRiskFactorModel('trig', params=params, ses=ses,
                                                               resids=pd.Series(np.zeros(10)))
        self._repository['a1c'] = NHANESLinearRiskFactorModel('a1c', params=params, ses=ses,
                                                              resids=pd.Series(np.zeros(10)))
        self._repository['hdl'] = NHANESLinearRiskFactorModel('hdl', params=params, ses=ses,
                                                              resids=pd.Series(np.zeros(10)))
        self._repository['totChol'] = NHANESLinearRiskFactorModel(
            'totChol', params=params, ses=ses, resids=pd.Series(np.zeros(10)))
        self._repository['bmi'] = NHANESLinearRiskFactorModel('bmi', params=params, ses=ses,
                                                              resids=pd.Series(np.zeros(10)))
        self._repository['waist'] = NHANESLinearRiskFactorModel('waist', params=params, ses=ses,
                                                                resids=pd.Series(np.zeros(10)))
        self._repository['anyPhysicalActivity'] = NHANESLinearRiskFactorModel('waist', params=params, ses=ses,
                                                                              resids=pd.Series(np.zeros(10)))
        self._repository['statin'] = NHANESLinearRiskFactorModel('statin', params=params, ses=ses,
                                                                 resids=pd.Series(np.zeros(10)))
        self._repository['antiHypertensiveCount'] = NHANESLinearRiskFactorModel('antiHypertensiveCount', params=params, ses=ses,
                                                                                resids=pd.Series(np.zeros(10)))
        self._repository['afib'] = NHANESLinearRiskFactorModel('afib', params=params, ses=ses,
                                                               resids=pd.Series(np.zeros(10)))
        self._repository['alcoholPerWeek'] = NHANESLinearRiskFactorModel('alcoholPerWeek', params=params, ses=ses,
                                                               resids=pd.Series(np.zeros(10)))
