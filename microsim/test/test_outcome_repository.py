from microsim.gender import NHANESGender
from microsim.person import Person
from microsim.outcome_model_type import OutcomeModelType
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.outcome_model_repository import OutcomeModelRepository, PersonRowWrapper
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe,
)
from microsim.test.helper import skip_if_quick_mode

import unittest


def initializeAfib(person):
    return None


@skip_if_quick_mode
class TestOutcomeRepository(unittest.TestCase):
    def setUp(self):
        self._white_male = Person(
            age=55,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,
            initializeAfib=initializeAfib,
        )

        self._black_male = Person(
            age=55,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=200,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,            
            creatinine = 0,
            initializeAfib=initializeAfib,
        )

        self._black_treated_male = Person(
            age=55,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=200,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=1,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,
            initializeAfib=initializeAfib,
        )

        self._white_female = Person(
            age=55,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,
            initializeAfib=initializeAfib,
        )

        self._black_female = Person(
            age=55,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,
            initializeAfib=initializeAfib,
        )

        self._population_dataframe = init_vectorized_population_dataframe(
            [
                self._black_female,
                self._black_male,
                self._black_treated_male,
            ],
            with_base_gcp=True,
        )
        self._outcome_model_repository = OutcomeModelRepository()

    def test_get_model_for_person(self):
        self.assertEqual(
            0.106501,
            self._outcome_model_repository.select_model_for_person(
                self._white_female, OutcomeModelType.CARDIOVASCULAR
            ).parameters["lagAge"],
        )
        self.assertEqual(
            0.106501,
            self._outcome_model_repository.select_model_for_person(
                self._black_female, OutcomeModelType.CARDIOVASCULAR
            ).parameters["lagAge"],
        )
        self.assertEqual(
            0.064200,
            self._outcome_model_repository.select_model_for_person(
                self._white_male, OutcomeModelType.CARDIOVASCULAR
            ).parameters["lagAge"],
        )
        self.assertEqual(
            0.064200,
            self._outcome_model_repository.select_model_for_person(
                self._black_male, OutcomeModelType.CARDIOVASCULAR
            ).parameters["lagAge"],
        )

    # this is testing whether our ASCVD model class directly reproduces the yadlowsky paper cases
    def test_calculate_actual_ten_year_risk_for_person(self):
        p1_data = self._population_dataframe.iloc[0]  # same data as `self._black_female`
        p1_model = self._outcome_model_repository.select_model_for_person(
            PersonRowWrapper(p1_data),
            OutcomeModelType.CARDIOVASCULAR,
        )
        p1_ten_year_risk = p1_model.transform_to_ten_year_risk(
            p1_model.get_one_year_linear_predictor(p1_data, vectorized=True)
        )
        self.assertAlmostEqual(0.017654, p1_ten_year_risk, delta=0.00001)

        # note that the reference value here is the corrected version of the
        # appendis table with the tot_chol/hdl ratio set to 4 for both the overall term and
        # the race interaction term
        p2_data = self._population_dataframe.iloc[1]  # same data as `self._black_male`
        p2_model = self._outcome_model_repository.select_model_for_person(
            PersonRowWrapper(p2_data),
            OutcomeModelType.CARDIOVASCULAR,
        )
        p2_ten_year_risk = p2_model.transform_to_ten_year_risk(
            p2_model.get_one_year_linear_predictor(p2_data, vectorized=True)
        )
        self.assertAlmostEqual(0.03476, p2_ten_year_risk, delta=0.00001)

    def test_approximate_one_year_risk_for_person(self):
        p1_data = self._population_dataframe.iloc[0]  # same data as `self._black_female`
        p1_one_year_risk = self._outcome_model_repository.get_risk_for_person(
            p1_data,
            OutcomeModelType.CARDIOVASCULAR,
            years=1,
            vectorized=True,
        )
        self.assertAlmostEqual(0.017654 / 10, p1_one_year_risk, delta=0.03)

        p2_data = self._population_dataframe.iloc[1]  # same data as `self._black_male`
        p2_one_year_risk = self._outcome_model_repository.get_risk_for_person(
            p2_data,
            OutcomeModelType.CARDIOVASCULAR,
            years=1,
            vectorized=True,
        )
        self.assertAlmostEqual(0.03476 / 10, p2_one_year_risk, delta=0.03)

    # details of risk worked out in example_treated_ascvd_scenario.xlsx
    def test_calculate_actual_ten_year_risk_for_treated_person(self):
        p3_data = self._population_dataframe.iloc[2]  # same data as `self._treated_black_male`

        model = self._outcome_model_repository.select_model_for_person(
            PersonRowWrapper(p3_data),
            OutcomeModelType.CARDIOVASCULAR,
        )
        actual_ten_year_risk = model.transform_to_ten_year_risk(
            model.get_one_year_linear_predictor(self._black_treated_male)
        )

        self.assertAlmostEqual(0.069810753, actual_ten_year_risk, delta=0.00001)

    def test_approximate_one_year_risk_for_person(self):
        p3_data = self._population_dataframe.iloc[2]  # same data as `self._treated_black_male`

        actual_one_year_risk = self._outcome_model_repository.get_risk_for_person(
            p3_data,
            OutcomeModelType.CARDIOVASCULAR,
            years=1,
            vectorized=True,
        )

        self.assertAlmostEqual(0.069810753 / 10, actual_one_year_risk, delta=0.03)
