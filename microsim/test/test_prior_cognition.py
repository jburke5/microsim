import unittest
import pandas as pd

from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.population.population_factory import PopulationFactory
from microsim.population.population_model_repository import PopulationRepositoryType
from microsim.outcomes.outcome import OutcomeType
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.modality import Modality
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.outcomes.cognition_outcome import CognitionOutcome
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository
from microsim.person.person_filter_factory import PersonFilterFactory


class TestMCIFilter(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame(
            {
                DynamicRiskFactorsType.AGE.value: 60,
                StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
                DynamicRiskFactorsType.SBP.value: 120,
                DynamicRiskFactorsType.DBP.value: 80,
                DynamicRiskFactorsType.A1C.value: 5.5,
                DynamicRiskFactorsType.HDL.value: 50,
                DynamicRiskFactorsType.TOT_CHOL.value: 200,
                DynamicRiskFactorsType.BMI.value: 25,
                DynamicRiskFactorsType.LDL.value: 90,
                DynamicRiskFactorsType.TRIG.value: 150,
                DynamicRiskFactorsType.WAIST.value: 45,
                DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                DefaultTreatmentsType.STATIN.value: 0,
                DynamicRiskFactorsType.CREATININE.value: 0.9,
                "name": "testPerson",
            },
            index=[0],
        )
        self.pf = PersonFilterFactory.get_person_filter(["noMCI"])
        self.filterFunction = self.pf.filters["person"]["noMCI"]

    def test_mci_filter_excludes_person_with_low_gcp(self):
        person = PersonFactory.get_nhanes_person(self.x.iloc[0], InitializationModelRepository())
        person._afib = [False]
        # MCI cutoff for age 60: 72.3182 - 0.2945*60 - 1.5*9.05 = 41.0732
        person._outcomes[OutcomeType.COGNITION] = []
        person.add_outcome(CognitionOutcome(False, True, 30))
        self.assertFalse(self.filterFunction(person))

    def test_mci_filter_keeps_person_with_normal_gcp(self):
        person = PersonFactory.get_nhanes_person(self.x.iloc[0], InitializationModelRepository())
        person._afib = [False]
        person._outcomes[OutcomeType.COGNITION] = []
        person.add_outcome(CognitionOutcome(False, True, 55))
        self.assertTrue(self.filterFunction(person))


class TestPriorCognitionNHANES(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame(
            {
                DynamicRiskFactorsType.AGE.value: 60,
                StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
                DynamicRiskFactorsType.SBP.value: 120,
                DynamicRiskFactorsType.DBP.value: 80,
                DynamicRiskFactorsType.A1C.value: 5.5,
                DynamicRiskFactorsType.HDL.value: 50,
                DynamicRiskFactorsType.TOT_CHOL.value: 200,
                DynamicRiskFactorsType.BMI.value: 25,
                DynamicRiskFactorsType.LDL.value: 90,
                DynamicRiskFactorsType.TRIG.value: 150,
                DynamicRiskFactorsType.WAIST.value: 45,
                DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                DefaultTreatmentsType.STATIN.value: 0,
                DynamicRiskFactorsType.CREATININE.value: 0.9,
                "name": "testPerson",
            },
            index=[0],
        )
        self.person = PersonFactory.get_nhanes_person(
            self.x.iloc[0],
            InitializationModelRepository(),
            outcomePrevalenceModelRepository=OutcomePrevalenceModelRepository(),
        )
        self.person._afib = [False]

    def test_nhanes_person_has_one_cognition_outcome_after_init(self):
        cognitionOutcomes = self.person._outcomes[OutcomeType.COGNITION]
        self.assertEqual(1, len(cognitionOutcomes))

    def test_nhanes_person_cognition_outcome_is_prior_to_sim(self):
        cognitionOutcomes = self.person._outcomes[OutcomeType.COGNITION]
        self.assertTrue(cognitionOutcomes[0][1].priorToSim)

    def test_baseline_gcp_raises_before_advance(self):
        with self.assertRaises(RuntimeError):
            _ = self.person._baselineGcp

    def test_baseline_gcp_returns_in_sim_gcp_after_advance(self):
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()._repository
        self.person.advance(
            1,
            popModelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
            popModelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
            popModelRepository[PopulationRepositoryType.OUTCOMES.value],
            None,
        )
        cognitionOutcomes = self.person._outcomes[OutcomeType.COGNITION]
        inSimCognition = [o for o in cognitionOutcomes if not o[1].priorToSim]
        self.assertEqual(self.person._baselineGcp, inSimCognition[0][1].gcp)

    def test_gcp_slope_nonzero_after_one_advance(self):
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()._repository
        self.person.advance(
            1,
            popModelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
            popModelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
            popModelRepository[PopulationRepositoryType.OUTCOMES.value],
            None,
        )
        self.assertNotEqual(0, self.person._gcpSlope)


class TestPriorCognitionKaiser(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame(
            {
                DynamicRiskFactorsType.AGE.value: 60,
                StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
                StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                StaticRiskFactorsType.MODALITY.value: Modality.MR.value,
                DynamicRiskFactorsType.SBP.value: 120,
                DynamicRiskFactorsType.DBP.value: 80,
                DynamicRiskFactorsType.A1C.value: 5.5,
                DynamicRiskFactorsType.HDL.value: 50,
                DynamicRiskFactorsType.TOT_CHOL.value: 200,
                DynamicRiskFactorsType.BMI.value: 25,
                DynamicRiskFactorsType.LDL.value: 90,
                DynamicRiskFactorsType.TRIG.value: 150,
                DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                DynamicRiskFactorsType.AFIB.value: False,
                DynamicRiskFactorsType.PVD.value: False,
                DynamicRiskFactorsType.CREATININE.value: 0.9,
                DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                DefaultTreatmentsType.STATIN.value: 0,
                "name": "testKaiserPerson",
            },
            index=[0],
        )
        self.person = PersonFactory.get_kaiser_person(self.x.iloc[0])

    def test_kaiser_person_has_one_cognition_outcome_after_init(self):
        cognitionOutcomes = self.person._outcomes[OutcomeType.COGNITION]
        self.assertEqual(1, len(cognitionOutcomes))

    def test_kaiser_person_cognition_outcome_is_prior_to_sim(self):
        cognitionOutcomes = self.person._outcomes[OutcomeType.COGNITION]
        self.assertTrue(cognitionOutcomes[0][1].priorToSim)

    def test_baseline_gcp_raises_before_advance(self):
        with self.assertRaises(RuntimeError):
            _ = self.person._baselineGcp

    def test_baseline_gcp_returns_in_sim_gcp_after_advance(self):
        popModelRepository = PopulationFactory.get_kaiser_population_model_repo()._repository
        self.person.advance(
            1,
            popModelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
            popModelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
            popModelRepository[PopulationRepositoryType.OUTCOMES.value],
            None,
        )
        cognitionOutcomes = self.person._outcomes[OutcomeType.COGNITION]
        inSimCognition = [o for o in cognitionOutcomes if not o[1].priorToSim]
        self.assertEqual(self.person._baselineGcp, inSimCognition[0][1].gcp)

    def test_gcp_slope_nonzero_after_one_advance(self):
        popModelRepository = PopulationFactory.get_kaiser_population_model_repo()._repository
        self.person.advance(
            1,
            popModelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
            popModelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
            popModelRepository[PopulationRepositoryType.OUTCOMES.value],
            None,
        )
        self.assertNotEqual(0, self.person._gcpSlope)


if __name__ == "__main__":
    unittest.main()
