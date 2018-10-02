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
            age=75, gender=0, race_ethnicity=1, sbp=140, dbp=80, a1c=6.5, hdl=50, chol=210)

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
            'chol': 0,
            'Intercept': 80,
        }

        ses = {
            'age': 0,
            'gender': 0,
            'raceEthnicity[T.2]': 0,
            'raceEthnicity[T.3]': 0,
            'raceEthnicity[T.4]': 0,
            'raceEthnicity[T.5]': 0,
            'sbp': 0,
            'dbp': 0,
            'a1c': 0,
            'hdl': 0,
            'chol': 0,
            'Intercept': 0,
        }

        self._risk_model_repository = {
            'sbp': LinearRiskFactorModel('sbp', params=params, ses=ses),
        }

    def test_sbp_model(self):
        self._test_person.advanceRiskFactors(self._risk_model_repository)
        expectedSBP = 75 * 1 + 140 * 0.5 + 80
        self.assertEqual(expectedSBP, self._test_person._sbp[-1])


if __name__ == "__main__":
    unittest.main()
