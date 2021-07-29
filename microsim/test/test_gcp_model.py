import unittest

from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.gcp_model_with_bp_treatment import GCPModel
from microsim.test.do_not_change_risk_factors_model_repository import (
    DoNotChangeRiskFactorsModelRepository,
)
from microsim.outcome_model_repository import OutcomeModelRepository


class AlwaysNegativeOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysNegativeOutcomeRepository, self).__init__()

    # override super to alays return a probability of each outcome as 0
    def get_risk_for_person(self, person, outcome, years=1):
        return 0


class TestGCPModel(unittest.TestCase):
    def initializeAfib(person):
        return None

    def setUp(self):
        self._test_case_one = Person(
            age=65 - 0.828576318 * 10,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120 + 0.45 * 10,
            dbp=80,
            # guessingon the centering standard for glucose...may have to check
            a1c=Person.convert_fasting_glucose_to_a1c(100 - 1.1 * 10),
            hdl=50,
            totChol=127 - 3.64 * 10,
            ldl=90,
            trig=150,
            bmi=26.6 + 15.30517532,
            waist=94 + 19.3,
            anyPhysicalActivity=1,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=1,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,        
            initializeAfib=TestGCPModel.initializeAfib,
        )

        self._test_case_two = Person(
            age=65 - 0.458555784 * 10,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120 + 0.3 * 10,
            dbp=80,
            # guessingon the centering standard for glucose...may have to check
            a1c=Person.convert_fasting_glucose_to_a1c(100 + 0.732746529 * 10),
            hdl=50,
            totChol=127 + 1.18 * 10,
            ldl=90,
            trig=150,
            bmi=26.6 + 0.419305619,
            waist=94 - 2.5,
            anyPhysicalActivity=1,
            education=Education.SOMEHIGHSCHOOL,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,        
            initializeAfib=TestGCPModel.initializeAfib,
        )

        self._test_case_three = Person(
            age=65 - 0.358692676 * 10,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120 + 2.3 * 10,
            dbp=80,
            # guessingon the centering standard for glucose...may have to check
            a1c=Person.convert_fasting_glucose_to_a1c(100 + 0.8893 * 10),
            hdl=50,
            totChol=127 + 4.7769 * 10,
            ldl=90,
            trig=150,
            bmi=26.6 + 2.717159247,
            waist=94 + 9,
            anyPhysicalActivity=1,
            education=Education.LESSTHANHIGHSCHOOL,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=1,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,
            initializeAfib=TestGCPModel.initializeAfib,
        )
        self._test_case_one._randomEffects["gcp"] = 0
        self._test_case_two._randomEffects["gcp"] = 0
        self._test_case_three._randomEffects["gcp"] = 0

    def test_baseline_gcp(self):
        # check that all of the random elements have been removed for testing...
        self.assertEqual(
            GCPModel().calc_linear_predictor(person=self._test_case_one, test=True),
            GCPModel().calc_linear_predictor(person=self._test_case_one, test=True),
        )

        # this is for the first person in teh spreadsheet (1569nomas)...comparator is GCP + backing out the cohort effect (NOMAS)
        self.assertAlmostEqual(
            64.45419405 - 2.7905,
            GCPModel().calc_linear_predictor(person=self._test_case_one, test=True),
            places=1,
        )

        # this is for the 2nd person in the spreadhsheet (204180409594cardia)...comparabor is GCP margin + backing out the cohort effect (cardia)
        self.assertAlmostEqual(
            50.01645213 + 1.3320,
            GCPModel().calc_linear_predictor(person=self._test_case_two, test=True),
            places=1,
        )

        # this is for the the first black person in the spreadsheet (J150483aric)...comparator is GCP + no cohort effect (ARIC)
        self.assertAlmostEqual(
            42.99471241,
            GCPModel().calc_linear_predictor(person=self._test_case_three, test=True),
            places=1,
        )

    def test_gcp_after_one_year(self):
        # this is for the first person in teh spreadsheet (1569nomas)...comparator is GCP + backing out the cohort effect (NOMAS)
        self._test_case_one.advance_year(
            DoNotChangeRiskFactorsModelRepository(), AlwaysNegativeOutcomeRepository()
        )
        # account for the difference between the actual mean systolic change
        self.assertAlmostEqual(
            64.28862964 - 2.7905,
            GCPModel().calc_linear_predictor(person=self._test_case_one, test=True),
            places=1,
        )

    def test_gcp_random_effect(self):
        self._test_case_one._randomEffects["gcp"] = 5
        self.assertAlmostEqual(
            64.45419405 - 2.7905 + 5,
            GCPModel().get_risk_for_person(person=self._test_case_one, years=1, test=True),
            places=1,
        )

    def test_gcp_random_effect_independent_per_person(self):
        expected_case_one_gcp = 66.66369405
        expected_case_two_gcp = 51.34845213
        self._test_case_one._randomEffects["gcp"] = 5
        gcp_model = GCPModel()

        actual_case_one_gcp = gcp_model.get_risk_for_person(self._test_case_one, test=True)
        actual_case_two_gcp = gcp_model.get_risk_for_person(self._test_case_two, test=True)

        self.assertAlmostEqual(expected_case_one_gcp, actual_case_one_gcp, places=1)
        self.assertAlmostEqual(expected_case_two_gcp, actual_case_two_gcp, places=1)
