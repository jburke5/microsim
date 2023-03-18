from microsim.person import Person
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.test.test_risk_model_repository import TestRiskModelRepository

import unittest
import numpy as np

def initializeAfib(person):
    return None


class TestNHANESLinearRiskFactorModel(unittest.TestCase):
    def setUp(self):
        self._test_person = Person(
            age=75,
            gender=0,
            raceEthnicity=1,
            sbp=140,
            dbp=80,
            a1c=6.5,
            hdl=50,
            totChol=210,
            ldl=90,
            trig=150,
            bmi=22,
            waist=50,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=1,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0,
            initializeAfib=initializeAfib,
            rng = np.random.default_rng(),
        )

        self._risk_model_repository = TestRiskModelRepository()

    def test_sbp_model(self):
        self._test_person.advance_risk_factors(self._risk_model_repository, rng = np.random.default_rng())
        expectedSBP = 75 * 1 + 140 * 0.5 + 80
        self.assertEqual(expectedSBP, self._test_person._sbp[-1])

    def test_upper_bounds(self):
        highBPPerson = Person(
            age=75,
            gender=0,
            raceEthnicity=1,
            sbp=500,
            ldl=90,
            trig=150,
            dbp=80,
            a1c=6.5,
            hdl=50,
            totChol=210,
            bmi=22,
            waist=50,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=1,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0,
            initializeAfib=initializeAfib,
            rng = np.random.default_rng(),
        )
        highBPPerson.advance_risk_factors(self._risk_model_repository, rng = np.random.default_rng())
        self.assertEqual(300, highBPPerson._sbp[-1])

        # TODO : write more tests — check the categorical variables and ensure
        # that all parameters are passed in or an error is thrown


if __name__ == "__main__":
    unittest.main()
