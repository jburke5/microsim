import unittest
import pandas as pd

from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.outcomes.diabetes_model import DiabetesModel
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.default_treatments.default_treatments import DefaultTreatmentsType


def _build_person(a1c):
    x = pd.DataFrame(
        {
            DynamicRiskFactorsType.AGE.value: 60,
            StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
            StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
            DynamicRiskFactorsType.SBP.value: 120,
            DynamicRiskFactorsType.DBP.value: 80,
            DynamicRiskFactorsType.A1C.value: a1c,
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
    return PersonFactory.get_nhanes_person(x.iloc[0], InitializationModelRepository())


class TestDiabetesPriorToSim(unittest.TestCase):
    def test_low_baseline_a1c_has_no_diabetes_outcome(self):
        person = _build_person(a1c=5.5)
        self.assertEqual(0, len(person._outcomes[OutcomeType.DIABETES]))


class TestDiabetesModel(unittest.TestCase):
    def test_emits_outcome_when_a1c_crosses_threshold(self):
        person = _build_person(a1c=5.5)
        person._a1c.append(7.0)
        outcome = DiabetesModel().get_next_outcome(person)
        self.assertIsNotNone(outcome)
        self.assertEqual(OutcomeType.DIABETES, outcome.type)
        self.assertFalse(outcome.fatal)
        self.assertFalse(outcome.priorToSim)

    def test_no_outcome_when_a1c_remains_below_threshold(self):
        person = _build_person(a1c=5.5)
        person._a1c.append(6.0)
        self.assertIsNone(DiabetesModel().get_next_outcome(person))

    def test_emits_outcome_for_priorToSim_diabetic(self):
        person = _build_person(a1c=7.0)
        outcome = DiabetesModel().get_next_outcome(person)
        self.assertIsNotNone(outcome)
        self.assertFalse(outcome.priorToSim)

    def test_continues_emitting_after_in_sim_event(self):
        person = _build_person(a1c=5.5)
        person._a1c.append(7.0)
        first = DiabetesModel().get_next_outcome(person)
        self.assertIsNotNone(first)
        person.add_outcome(first)

        person._a1c.append(7.5)
        self.assertIsNotNone(DiabetesModel().get_next_outcome(person))

    def test_continues_emitting_independent_of_a1c_once_outcome_exists(self):
        # Person never had high A1C, but a diabetes outcome is present (e.g. via priorToSim seed).
        # Once the outcome exists, the model must emit regardless of A1C.
        person = _build_person(a1c=5.0)
        self.assertEqual(0, len(person._outcomes[OutcomeType.DIABETES]))
        person._outcomes[OutcomeType.DIABETES].append(
            (None, Outcome(OutcomeType.DIABETES, False, priorToSim=True))
        )
        self.assertFalse(person.has_diabetes())
        self.assertIsNotNone(DiabetesModel().get_next_outcome(person))


class TestDiabetesRegistration(unittest.TestCase):
    def test_diabetes_in_outcome_model_repository(self):
        omr = OutcomeModelRepository()
        self.assertIn(OutcomeType.DIABETES, omr._repository)


if __name__ == "__main__":
    unittest.main()
