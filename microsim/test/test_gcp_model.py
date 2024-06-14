import unittest
import pandas as pd

from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.gcp_model import GCPModel
from microsim.test.do_not_change_risk_factors_model_repository import (
    DoNotChangeRiskFactorsModelRepository,
)
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.population_factory import PopulationFactory
from microsim.person_factory import PersonFactory
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.treatment import DefaultTreatmentsType

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
 
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        self.x_test_case_one = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 65 - 0.828576318 * 10,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120 + 0.45 * 10,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100 - 1.1 * 10),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 127 - 3.64 * 10,
                               DynamicRiskFactorsType.BMI.value: 26.6 + 15.30517532,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94 + 19.3,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 1,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_one"}, index=[0])
        self._test_case_one = PersonFactory.get_nhanes_person(self.x_test_case_one.iloc[0], initializationModelRepository)
        self._test_case_one._afib = [False]

        self.x_test_case_two = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 65 - 0.458555784 * 10,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120 + 0.3 * 10,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100 + 0.732746529 * 10),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 127 + 1.18 * 10,
                               DynamicRiskFactorsType.BMI.value: 26.6 + 0.419305619,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94 -2.5,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.SOMEHIGHSCHOOL.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_two"}, index=[0])
        self._test_case_two = PersonFactory.get_nhanes_person(self.x_test_case_two.iloc[0], initializationModelRepository)
        self._test_case_two._afib = [False]

        self.x_test_case_three = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 65 - 0.358692676 * 10,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120 + 2.3 * 10,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100 + 0.8893 * 10),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 127 + 4.7769 * 10,
                               DynamicRiskFactorsType.BMI.value: 26.6 + 2.717159247,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94 + 9,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.LESSTHANHIGHSCHOOL.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 1,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "test_case_three"}, index=[0])
        self._test_case_three = PersonFactory.get_nhanes_person(self.x_test_case_three.iloc[0], initializationModelRepository)
        self._test_case_three._afib = [False]

        self._test_case_one._randomEffects["gcp"] = 0
        self._test_case_two._randomEffects["gcp"] = 0
        self._test_case_three._randomEffects["gcp"] = 0

    def test_baseline_gcp(self):
        # check that all of the random elements have been removed for testing...
        self.assertEqual(
            GCPModel().get_risk_for_person(person=self._test_case_one, test=True),
            GCPModel().get_risk_for_person(person=self._test_case_one, test=True),
        )

        # this is for the first person in teh spreadsheet (1569nomas)...comparator is GCP + backing out the cohort effect (NOMAS)
        self.assertAlmostEqual(
            64.45419405 - 2.7905,
            GCPModel().get_risk_for_person(person=self._test_case_one, test=True),
            places=1,
        )

        # this is for the 2nd person in the spreadhsheet (204180409594cardia)...comparabor is GCP margin + backing out the cohort effect (cardia)
        self.assertAlmostEqual(
            50.01645213 + 1.3320,
            GCPModel().get_risk_for_person(person=self._test_case_two, test=True),
            places=1,
        )

        # this is for the the first black person in the spreadsheet (J150483aric)...comparator is GCP + no cohort effect (ARIC)
        self.assertAlmostEqual(
            42.99471241,
            GCPModel().get_risk_for_person(person=self._test_case_three, test=True),
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

if __name__ == "__main__":
    unittest.main()
