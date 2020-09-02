import unittest
import numpy as np

from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.dementia_model import DementiaModel
from microsim.test.do_not_change_risk_factors_model_repository import DoNotChangeRiskFactorsModelRepository
from microsim.outcome_model_repository import OutcomeModelRepository


class AlwaysNegativeOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysNegativeOutcomeRepository, self).__init__()

    # override super to alays return a probability of each outcome as 0
    def get_risk_for_person(self, person, outcome, years=1):
        return 0

    def get_gcp(self, person):
        return person._gcp[-1]


class TestDementiaModel(unittest.TestCase):
    def initializeAfib(person):
        return None

    def setUp(self):
        # test case 1 in rep_gdta.
        self._test_case_one = Person(
            age=40,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120,
            dbp=80,
            # guessingon the centering standard for glucose...may have to check
            a1c=Person.convert_fasting_glucose_to_a1c(100),
            hdl=50,
            totChol=150,
            ldl=90,
            trig=150,
            bmi=26.6,
            waist=94,
            anyPhysicalActivity=1,
            education=Education.LESSTHANHIGHSCHOOL,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            initializeAfib=TestDementiaModel.initializeAfib)
        self._test_case_one._gcp[0] = 25
        # GCP slope is zero
        self._test_case_one._gcp.append(self._test_case_one._gcp[0])

        # test case 71 in rep_gdta.
        self._test_case_two = Person(
            age=80,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120,
            dbp=80,
            # guessingon the centering standard for glucose...may have to check
            a1c=Person.convert_fasting_glucose_to_a1c(100),
            hdl=50,
            totChol=150,
            ldl=90,
            trig=150,
            bmi=26.6,
            waist=94,
            anyPhysicalActivity=1,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            initializeAfib=TestDementiaModel.initializeAfib)
        self._test_case_two._gcp[0] = 75
        self._test_case_two._gcp.append(self._test_case_two._gcp[0])

        # test case 72 in rep_gdta.
        self._test_case_three = Person(
            age=80,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120,
            dbp=80,
            # guessingon the centering standard for glucose...may have to check
            a1c=Person.convert_fasting_glucose_to_a1c(100),
            hdl=50,
            totChol=150,
            ldl=90,
            trig=150,
            bmi=26.6,
            waist=94,
            anyPhysicalActivity=1,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            initializeAfib=TestDementiaModel.initializeAfib)
        self._test_case_three._gcp[0] = 75
        self._test_case_three._gcp.append(self._test_case_two._gcp[0])

    def test_dementia_after_one_year(self):
        self.assertAlmostEqual(-9.990598486, DementiaModel().linear_predictor(person=self._test_case_one), places=5)
        self.assertAlmostEqual(5.19E-05, DementiaModel().get_risk_for_person(person=self._test_case_one, years=1), places=5)

    def test_dementia_after_one_year_person_two(self):
        self.assertAlmostEqual(-5.804540672, DementiaModel().linear_predictor(person=self._test_case_two), places=5)
        self.assertAlmostEqual(0.003411382, DementiaModel().get_risk_for_person(person=self._test_case_two, years=1), places=5)

    def test_dementia_after_one_year_person_three(self):
        self.assertAlmostEqual(-6.018035196, DementiaModel().linear_predictor(person=self._test_case_three), places=5)
        self.assertAlmostEqual(0.002755566, DementiaModel().get_risk_for_person(person=self._test_case_three, years=1), places=5)
