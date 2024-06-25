from microsim.gender import NHANESGender
from microsim.person import Person
from microsim.race_ethnicity import RaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe,
)
from microsim.treatment import DefaultTreatmentsType
from microsim.population_factory import PopulationFactory
from microsim.person_factory import PersonFactory
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.outcome import OutcomeType

import unittest
import numpy as np
import pandas as pd

class TestOutcomeRepository(unittest.TestCase):
    def setUp(self):
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        self.x_white_male = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 55,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 213,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 34,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "white_male"}, index=[0])
        self._white_male = PersonFactory.get_nhanes_person(self.x_white_male.iloc[0], initializationModelRepository)
        self._white_male._afib = [False]

        self.x_black_male = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 55,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 200,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 34,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "black_male"}, index=[0])
        self._black_male = PersonFactory.get_nhanes_person(self.x_black_male.iloc[0], initializationModelRepository)
        self._black_male._afib = [False]

        self.x_treated_black_male = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 55,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 200,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 34,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 1,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "black_treated_male"}, index=[0])
        self._treated_black_male = PersonFactory.get_nhanes_person(self.x_treated_black_male.iloc[0], initializationModelRepository)
        self._treated_black_male._afib = [False]

        self.x_white_female = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 55,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 213,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 34,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "white_female"}, index=[0])
        self._white_female = PersonFactory.get_nhanes_person(self.x_white_female.iloc[0], initializationModelRepository)
        self._white_female._afib = [False]

        self.x_black_female = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 55,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 213,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 34,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "black_female"}, index=[0])
        self._black_female = PersonFactory.get_nhanes_person(self.x_black_female.iloc[0], initializationModelRepository)
        self._black_female._afib = [False]

        self._outcome_model_repository = OutcomeModelRepository()

    def test_get_model_for_person(self):
        self.assertEqual(
            0.106501,
            self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._white_female).parameters["lagAge"]
        )
        self.assertEqual(
            0.106501,
            self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._black_female).parameters["lagAge"]
        )
        self.assertEqual(
            0.064200,
            self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._white_male).parameters["lagAge"]
        )
        self.assertEqual(
            0.064200,
            self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._black_male).parameters["lagAge"]
        )

    # this is testing whether our ASCVD model class directly reproduces the yadlowsky paper cases
    def test_calculate_actual_ten_year_risk_for_person(self):
        p1_model = self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._black_female)
        p1_ten_year_risk = p1_model.transform_to_ten_year_risk(
            p1_model.get_one_year_linear_predictor(self._black_female)
        )
        self.assertAlmostEqual(0.017654, p1_ten_year_risk, delta=0.00001)

        # note that the reference value here is the corrected version of the
        # appendis table with the tot_chol/hdl ratio set to 4 for both the overall term and
        # the race interaction term
        p2_model = self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._black_male)
        p2_ten_year_risk = p2_model.transform_to_ten_year_risk(
            p2_model.get_one_year_linear_predictor(self._black_male)
        )
        self.assertAlmostEqual(0.03476, p2_ten_year_risk, delta=0.00001)

    def test_approximate_one_year_risk_for_person(self):

        p1_model = self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._black_female)
        p1_one_year_risk = p1_model.get_risk_for_person(self._black_female, years=1)
        self.assertAlmostEqual(0.017654 / 10, p1_one_year_risk, delta=0.03)

        p2_model = self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._black_male)
        p2_one_year_risk = p2_model.get_risk_for_person(self._black_male, years=1)
        self.assertAlmostEqual(0.03476 / 10, p2_one_year_risk, delta=0.03)

    # details of risk worked out in example_treated_ascvd_scenario.xlsx
    def test_calculate_actual_ten_year_risk_for_treated_person(self):
        model = self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._treated_black_male)
        actual_ten_year_risk = model.transform_to_ten_year_risk(
           model.get_one_year_linear_predictor(self._treated_black_male)
        )
        self.assertAlmostEqual(0.069810753, actual_ten_year_risk, delta=0.00001)

    def test_approximate_one_year_risk_for_person(self):
        model = self._outcome_model_repository._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(self._treated_black_male)
        actual_one_year_risk = model.get_risk_for_person(self._treated_black_male, years=1)
        self.assertAlmostEqual(0.069810753 / 10, actual_one_year_risk, delta=0.03)

if __name__ == "__main__":
    unittest.main()
