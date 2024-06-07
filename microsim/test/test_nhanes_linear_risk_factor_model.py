from microsim.person import Person
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.population_factory import PopulationFactory
from microsim.treatment import DefaultTreatmentsType
from microsim.person_factory import PersonFactory
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity

import unittest
import numpy as np
import pandas as pd

def initializeAfib(person):
    return None


class TestNHANESLinearRiskFactorModel(unittest.TestCase):
    def setUp(self):
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        self.x_test_person = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 75,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.MEXICAN_AMERICAN.value,
                               DynamicRiskFactorsType.SBP.value: 140,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 210,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 50,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.FORMER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_person"}, index=[0])
        self._test_person = PersonFactory.get_nhanes_person(self.x_test_person.iloc[0], initializationModelRepository)
        self._test_person._afib = [False]

        self._risk_model_repository = TestRiskModelRepository()

    def test_sbp_model(self):
        self._test_person.advance_risk_factors(self._risk_model_repository)
        expectedSBP = 75 * 1 + 140 * 0.5 + 80
        self.assertEqual(expectedSBP, self._test_person._sbp[-1])

    def test_upper_bounds(self):
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()
        x_highBPPerson = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 75,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.MEXICAN_AMERICAN.value,
                               DynamicRiskFactorsType.SBP.value: 500,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 210,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 50,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.FORMER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "highBPPerson"}, index=[0])
        highBPPerson = PersonFactory.get_nhanes_person(x_highBPPerson.iloc[0], initializationModelRepository)
        highBPPerson._afib = [False]

        highBPPerson.advance_risk_factors(self._risk_model_repository)
        self.assertEqual(297, highBPPerson._sbp[-1])

        # TODO : write more tests — check the categorical variables and ensure
        # that all parameters are passed in or an error is thrown


if __name__ == "__main__":
    unittest.main()
