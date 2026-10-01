"""Tests for the outcome-attribute-slicing functions in person.py:
get_wave_at_last_outcome, get_attr_prior_first_outcome_in_sim,
get_attr_prior_last_outcome, get_attr_since_last_outcome and their mean/median wrappers.

priorToSim outcomes carry age=None (see Person.add_outcome), so these functions must
detect prior-to-sim outcomes from the outcome records, not from the stored age.
The *PriorToSimOnly tests pin the behavior that used to be broken: a person whose only
outcome was prior to sim was misclassified as never having had the outcome.
"""

import unittest
import pandas as pd

from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.stroke_outcome import StrokeOutcome
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.default_treatments.default_treatments import DefaultTreatmentsType


def _build_person():
    x = pd.DataFrame({
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
        "name": "testPerson"}, index=[0])
    person = PersonFactory.get_nhanes_person(x.iloc[0], InitializationModelRepository())
    #simulate a person advanced 3 waves without running any models
    person._age = [60, 61, 62, 63]
    person._sbp = [120, 125, 130, 135]
    person._waveCompleted = 3
    return person


def _add_priorToSim_stroke(person):
    person._outcomes[OutcomeType.STROKE].append(
        (None, StrokeOutcome(False, None, None, None, priorToSim=True)))


def _add_in_sim_stroke(person, age):
    person._outcomes[OutcomeType.STROKE].append(
        (age, StrokeOutcome(False, None, None, None, priorToSim=False)))


class TestGetWaveAtLastOutcome(unittest.TestCase):

    def test_never_had_outcome(self):
        person = _build_person()
        self.assertIsNone(person.get_wave_at_last_outcome(OutcomeType.STROKE))

    def test_priorToSim_only(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertIsNone(person.get_wave_at_last_outcome(OutcomeType.STROKE))

    def test_in_sim_outcome(self):
        person = _build_person()
        _add_in_sim_stroke(person, age=62)
        self.assertEqual(2, person.get_wave_at_last_outcome(OutcomeType.STROKE))

    def test_priorToSim_and_in_sim_outcome(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=62)
        self.assertEqual(2, person.get_wave_at_last_outcome(OutcomeType.STROKE))


class TestGetAttrPriorFirstOutcomeInSim(unittest.TestCase):

    def test_never_had_outcome(self):
        person = _build_person()
        self.assertIsNone(person.get_attr_prior_first_outcome_in_sim("_sbp", OutcomeType.STROKE))

    def test_priorToSim_only(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertIsNone(person.get_attr_prior_first_outcome_in_sim("_sbp", OutcomeType.STROKE))

    def test_in_sim_outcome(self):
        person = _build_person()
        _add_in_sim_stroke(person, age=62)
        self.assertEqual([120, 125], person.get_attr_prior_first_outcome_in_sim("_sbp", OutcomeType.STROKE))

    def test_priorToSim_and_in_sim_outcome(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=62)
        self.assertEqual([120, 125], person.get_attr_prior_first_outcome_in_sim("_sbp", OutcomeType.STROKE))


class TestGetAttrPriorLastOutcome(unittest.TestCase):

    def test_never_had_outcome(self):
        person = _build_person()
        self.assertIsNone(person.get_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))

    def test_priorToSim_only_returns_baseline(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertEqual(120, person.get_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))

    def test_in_sim_outcome(self):
        person = _build_person()
        _add_in_sim_stroke(person, age=62)
        self.assertEqual([120, 125], person.get_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))

    def test_priorToSim_and_in_sim_outcome(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=62)
        self.assertEqual([120, 125], person.get_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))

    def test_two_in_sim_outcomes_uses_last(self):
        person = _build_person()
        _add_in_sim_stroke(person, age=61)
        _add_in_sim_stroke(person, age=63)
        self.assertEqual([120, 125, 130], person.get_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))


class TestGetAttrSinceLastOutcome(unittest.TestCase):

    def test_never_had_outcome(self):
        person = _build_person()
        self.assertIsNone(person.get_attr_since_last_outcome("_sbp", OutcomeType.STROKE))

    def test_priorToSim_only_returns_entire_list(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertEqual([120, 125, 130, 135], person.get_attr_since_last_outcome("_sbp", OutcomeType.STROKE))

    def test_in_sim_outcome(self):
        person = _build_person()
        _add_in_sim_stroke(person, age=62)
        self.assertEqual([130, 135], person.get_attr_since_last_outcome("_sbp", OutcomeType.STROKE))

    def test_priorToSim_and_in_sim_outcome(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=62)
        self.assertEqual([130, 135], person.get_attr_since_last_outcome("_sbp", OutcomeType.STROKE))


class TestMeanMedianWrappers(unittest.TestCase):

    def test_median_attr_prior_last_outcome_priorToSim_only(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertEqual(120, person.get_median_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))

    def test_mean_attr_prior_last_outcome_in_sim(self):
        person = _build_person()
        _add_in_sim_stroke(person, age=62)
        self.assertAlmostEqual(122.5, person.get_mean_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))

    def test_mean_attr_since_last_outcome_priorToSim_only(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertAlmostEqual(127.5, person.get_mean_attr_since_last_outcome("_sbp", OutcomeType.STROKE))

    def test_wrappers_return_none_when_no_outcome(self):
        person = _build_person()
        self.assertIsNone(person.get_median_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))
        self.assertIsNone(person.get_mean_attr_prior_last_outcome("_sbp", OutcomeType.STROKE))
        self.assertIsNone(person.get_mean_attr_since_last_outcome("_sbp", OutcomeType.STROKE))


if __name__ == "__main__":
    unittest.main()
