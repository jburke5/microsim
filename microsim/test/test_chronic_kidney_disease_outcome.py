import unittest
import pandas as pd

from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.outcomes.chronic_kidney_disease_model import ChronicKidneyDiseaseModel
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.default_treatments.default_treatments import DefaultTreatmentsType


def _build_person(creatinine):
    x = pd.DataFrame(
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
            DynamicRiskFactorsType.CREATININE.value: creatinine,
            "name": "testPerson",
        },
        index=[0],
    )
    return PersonFactory.get_nhanes_person(x.iloc[0], InitializationModelRepository())


# Creatinine values calibrated for a 60yo non-Hispanic-white male:
# 0.9 -> GFR ~92 (not CKD), 1.5 -> GFR ~50 (CKD).
NON_CKD_CREATININE = 0.9
CKD_CREATININE = 1.5


class TestCKDPersonInit(unittest.TestCase):
    def test_no_outcome_at_init_regardless_of_gfr(self):
        # priorToSim is intentionally not seeded for this outcome.
        for creatinine in [NON_CKD_CREATININE, CKD_CREATININE]:
            person = _build_person(creatinine)
            self.assertEqual(0, len(person._outcomes[OutcomeType.CHRONIC_KIDNEY_DISEASE]))


class TestCKDModel(unittest.TestCase):
    def test_emits_outcome_when_gfr_below_60(self):
        person = _build_person(CKD_CREATININE)
        self.assertTrue(person._current_ckd)
        outcome = ChronicKidneyDiseaseModel().get_next_outcome(person)
        self.assertIsNotNone(outcome)
        self.assertEqual(OutcomeType.CHRONIC_KIDNEY_DISEASE, outcome.type)
        self.assertFalse(outcome.fatal)
        self.assertFalse(outcome.priorToSim)

    def test_no_outcome_when_gfr_at_or_above_60(self):
        person = _build_person(NON_CKD_CREATININE)
        self.assertFalse(person._current_ckd)
        self.assertIsNone(ChronicKidneyDiseaseModel().get_next_outcome(person))

    def test_emits_outcome_when_creatinine_rises_mid_sim(self):
        person = _build_person(NON_CKD_CREATININE)
        person._creatinine.append(CKD_CREATININE)
        self.assertIsNotNone(ChronicKidneyDiseaseModel().get_next_outcome(person))

    def test_continues_emitting_after_in_sim_event(self):
        person = _build_person(CKD_CREATININE)
        first = ChronicKidneyDiseaseModel().get_next_outcome(person)
        self.assertIsNotNone(first)
        person.add_outcome(first)

        # Next wave: creatinine recovers to non-CKD level, but outcome already recorded.
        person._creatinine.append(NON_CKD_CREATININE)
        self.assertFalse(person._current_ckd)
        self.assertIsNotNone(ChronicKidneyDiseaseModel().get_next_outcome(person))

    def test_continues_emitting_independent_of_gfr_once_outcome_exists(self):
        # Person never had low GFR, but a CKD outcome is present. Once the outcome exists,
        # the model must emit regardless of current GFR.
        person = _build_person(NON_CKD_CREATININE)
        self.assertFalse(person._current_ckd)
        person._outcomes[OutcomeType.CHRONIC_KIDNEY_DISEASE].append(
            (60, Outcome(OutcomeType.CHRONIC_KIDNEY_DISEASE, False))
        )
        self.assertIsNotNone(ChronicKidneyDiseaseModel().get_next_outcome(person))


class TestCKDRegistration(unittest.TestCase):
    def test_registered_in_outcome_model_repository(self):
        omr = OutcomeModelRepository()
        self.assertIn(OutcomeType.CHRONIC_KIDNEY_DISEASE, omr._repository)


if __name__ == "__main__":
    unittest.main()
