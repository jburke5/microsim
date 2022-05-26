import unittest
import numpy as np

from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.dementia_model import DementiaModel
from microsim.test.do_not_change_risk_factors_model_repository import (
    DoNotChangeRiskFactorsModelRepository,
)
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.initialization_repository import InitializationRepository
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe,
)


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
        # 2740200061fos
        self._test_case_one = Person(
            age=54.060233,
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
            education=Education.HIGHSCHOOLGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0,
            initializeAfib=TestDementiaModel.initializeAfib,
            initializationRepository=InitializationRepository(),
        )
        self._test_case_one._gcp[0] = 58.68
        self._test_case_one._gcp.append(self._test_case_one._gcp[0] - 1.1078128)

        # 2740201178fos
        self._test_case_two = Person(
            age=34.504449,
            gender=NHANESGender.MALE,
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
            education=Education.SOMECOLLEGE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0,
            initializeAfib=TestDementiaModel.initializeAfib,
            initializationRepository=InitializationRepository(),
        )
        self._test_case_two._gcp[0] = 58.68
        self._test_case_two._gcp.append(self._test_case_two._gcp[0] - 1.7339989)

        self._test_case_one_parameteric = Person(
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
            creatinine=0,
            initializeAfib=TestDementiaModel.initializeAfib,
            initializationRepository=InitializationRepository(),
        )
        self._test_case_one_parameteric._gcp[0] = 25
        # GCP slope is zero
        self._test_case_one_parameteric._gcp.append(self._test_case_one._gcp[0])

        # test case 71 in rep_gdta.
        self._test_case_two_parametric = Person(
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
            creatinine=0,
            initializeAfib=TestDementiaModel.initializeAfib,
            initializationRepository=InitializationRepository(),
        )
        self._test_case_two_parametric._gcp[0] = 75
        self._test_case_two_parametric._gcp.append(self._test_case_two._gcp[0])

        # test case 72 in rep_gdta.
        self._test_case_three_parametric = Person(
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
            creatinine=0,
            initializeAfib=TestDementiaModel.initializeAfib,
            initializationRepository=InitializationRepository(),
        )
        self._test_case_three_parametric._gcp[0] = 75
        self._test_case_three_parametric._gcp.append(self._test_case_two._gcp[0])

        self._population_dataframe = init_vectorized_population_dataframe(
            [
                self._test_case_one,
                self._test_case_two,
            ]
        )

    def test_dementia_after_one_year(self):
        p1_data = self._population_dataframe.iloc[0]

        actual_risk = DementiaModel().linear_predictor_vectorized(p1_data)

        self.assertAlmostEqual(1.115571, actual_risk, places=5)

    def test_dementia_after_one_year_person_two(self):
        p2_data = self._population_dataframe.iloc[1]

        actual_risk = DementiaModel().linear_predictor_vectorized(p2_data)

        self.assertAlmostEqual(-1.122424, actual_risk, places=5)
