import unittest
import numpy as np
import pandas as pd

from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import RaceEthnicity
from microsim.dementia_model import DementiaModel
from microsim.test.do_not_change_risk_factors_model_repository import (
    DoNotChangeRiskFactorsModelRepository,
)
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.initialization_repository import InitializationRepository
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe,
)
from microsim.treatment import DefaultTreatmentsType
from microsim.population_factory import PopulationFactory
from microsim.person_factory import PersonFactory
from microsim.dementia_model_repository import DementiaModelRepository
from microsim.cv_model_repository import CVModelRepository
from microsim.person_filter import PersonFilter
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.population_model_repository import PopulationRepositoryType
from microsim.cognition_outcome import CognitionOutcome
from microsim.outcome import OutcomeType

class TestDementiaModel(unittest.TestCase):

    def setUp(self):
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        # 2740200061fos
        self.x_test_case_one = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 54.06023,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.HIGHSCHOOLGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_one"}, index=[0])
        self._test_case_one = PersonFactory.get_nhanes_person(self.x_test_case_one.iloc[0], initializationModelRepository)
        self._test_case_one._afib = [False]
        #I initially used to advance the person to get an initial cognition outcome to get the initial gcp but then 
        #I realized I can just add the cognition outcomes
        #self._test_case_one.advance(1, popModelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
        #                       popModelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
        #                       popModelRepository[PopulationRepositoryType.OUTCOMES.value],
        #                       None)
        #self._test_case_one._outcomes[OutcomeType.COGNITION][0][1].gcp = 58.68
        self._test_case_one.add_outcome(CognitionOutcome(False, False, 58.68))
        self._test_case_one.add_outcome(CognitionOutcome(False, False, 58.68 - 1.1078128))

        # 2740201178fos
        self.x_test_case_two = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 34.504449,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.SOMECOLLEGE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_two"}, index=[0])
        self._test_case_two = PersonFactory.get_nhanes_person(self.x_test_case_two.iloc[0], initializationModelRepository)
        self._test_case_two._afib = [False]
        #self._test_case_two.advance(1, popModelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
        #                       popModelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
        #                       popModelRepository[PopulationRepositoryType.OUTCOMES.value],
        #                       None)
        #self._test_case_two._outcomes[OutcomeType.COGNITION][0][1].gcp = 58.68
        self._test_case_two.add_outcome(CognitionOutcome(False, False, 58.68 ))
        self._test_case_two.add_outcome(CognitionOutcome(False, False, 58.68 - 1.7339989))

        self.x_test_case_one_parametric = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 40,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.LESSTHANHIGHSCHOOL.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_one_parametric"}, index=[0])
        self._test_case_one_parametric = PersonFactory.get_nhanes_person(self.x_test_case_one_parametric.iloc[0], initializationModelRepository)
        self._test_case_one_parametric._afib = [False] 
        #self._test_case_one_parametric._gcp[0] = 25
        # GCP slope is zero
        #self._test_case_one_parametric._gcp.append(self._test_case_one_parametric._gcp[0])
        self._test_case_one_parametric.add_outcome(CognitionOutcome(False, False, 25 ))
        self._test_case_one_parametric.add_outcome(CognitionOutcome(False, False, 25 ))

        # test case 71 in rep_gdta.
        self.x_test_case_two_parametric = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 80,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_two_parametric"}, index=[0])
        self._test_case_two_parametric = PersonFactory.get_nhanes_person(self.x_test_case_two_parametric.iloc[0], initializationModelRepository)
        self._test_case_two_parametric._afib = [False]
        #self._test_case_two_parametric._gcp[0] = 75
        #self._test_case_two_parametric._gcp.append(self._test_case_two._gcp[0])
        #I cannot tell if we need to use test case two initial gcp here or a 0 gcp slope
        self._test_case_two_parametric.add_outcome(CognitionOutcome(False, False, 75 ))
        self._test_case_two_parametric.add_outcome(CognitionOutcome(False, False, 75 ))

        # test case 72 in rep_gdta.
        self.x_test_case_three_parametric = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 80,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_three_parametric"}, index=[0])
        self._test_case_three_parametric = PersonFactory.get_nhanes_person(self.x_test_case_three_parametric.iloc[0], initializationModelRepository)
        self._test_case_three_parametric._afib = [False]
        #self._test_case_three_parametric._gcp[0] = 75
        #self._test_case_three_parametric._gcp.append(self._test_case_two._gcp[0])
        self._test_case_three_parametric.add_outcome(CognitionOutcome(False, False, 75))
        #I cannot tell if this was supposed to be a 0 slope or really use test case two initial GCP
        self._test_case_three_parametric.add_outcome(CognitionOutcome(False, False, 75))

    def test_dementia_after_one_year(self):
        x = self._test_case_one
        actual_risk = DementiaModelRepository().select_outcome_model_for_person(x).linear_predictor(x)
        self.assertAlmostEqual(1.115571, actual_risk, places=5)

    def test_dementia_after_one_year_person_two(self):
        x = self._test_case_two
        actual_risk = DementiaModelRepository().select_outcome_model_for_person(x).linear_predictor(x)
        self.assertAlmostEqual(-1.122424, actual_risk, places=5)
       
if __name__ == "__main__":
    unittest.main()
