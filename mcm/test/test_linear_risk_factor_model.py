from mcm.population import NHANESDirectSamplePopulation
from mcm.linear_risk_factor_model import LinearRiskFactorModel
from mcm.person import Person
from statsmodels.regression.linear_model import OLSResults

import unittest
import pandas as pd
import numpy as np


class TestLinearRiskFactorModel(unittest.TestCase):
    def setUp(self):
        self._test_person = Person(
            age=75, gender=0, race_ethnicity=1, sbp=140, dbp=80, a1c=6.5, hdl=50, tot_chol=210,
            bmi=22, smoking_status=1)

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
            'tot_chol': 0,
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
            'tot_chol': 0,
            'Intercept': 0,
        }

        self._risk_model_repository = {
            'sbp': LinearRiskFactorModel('sbp', params=params, ses=ses,
                                         resids=pd.Series(np.zeros(10))),
            'dbp': LinearRiskFactorModel('dbp', params=params, ses=ses,
                                         resids=pd.Series(np.zeros(10))),
            'a1c': LinearRiskFactorModel('a1c', params=params, ses=ses,
                                         resids=pd.Series(np.zeros(10))),
            'hdl': LinearRiskFactorModel('hdl', params=params, ses=ses,
                                         resids=pd.Series(np.zeros(10))),
            'tot_chol': LinearRiskFactorModel('tot_chol', params=params, ses=ses,
                                              resids=pd.Series(np.zeros(10))),
            'bmi': LinearRiskFactorModel('bmi', params=params, ses=ses,
                                         resids=pd.Series(np.zeros(10))),
        }

    def test_sbp_model(self):
        self._test_person.advance_risk_factors(self._risk_model_repository)
        expectedSBP = 75 * 1 + 140 * 0.5 + 80
        self.assertEqual(expectedSBP, self._test_person._sbp[-1])

    def test_upper_bounds(self):
        highBPPerson = Person(age=75, gender=0, race_ethnicity=1, sbp=500,
                              dbp=80, a1c=6.5, hdl=50, tot_chol=210, bmi=22, smoking_status=1)
        highBPPerson.advance_risk_factors(self._risk_model_repository)
        self.assertEqual(300, highBPPerson._sbp[-1])

        # TODO : write more tests — check the categorical variables and ensure that all parameters are passed in or an error is thrown


if __name__ == "__main__":
    unittest.main()
