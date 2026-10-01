"""Regression tests for None-age handling in functions that consume outcome ages.

Per the priorToSim convention, outcomes flagged priorToSim=True carry an age of None
(see Person.add_outcome). These tests pin down how every age-consuming function in
person.py reacts to that sentinel.

Some tests intentionally fail today — they surface bugs that the None convention
exposed and that need separate fixes. Those tests are tagged in their docstrings.
"""

import unittest
import pandas as pd

from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.outcomes.outcome import Outcome, OutcomeType
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
    return PersonFactory.get_nhanes_person(x.iloc[0], InitializationModelRepository())


def _add_priorToSim_stroke(person):
    person._outcomes[OutcomeType.STROKE].append(
        (None, StrokeOutcome(False, None, None, None, priorToSim=True)))


def _add_in_sim_stroke(person, age):
    person._outcomes[OutcomeType.STROKE].append(
        (age, StrokeOutcome(False, None, None, None, priorToSim=False)))


# --- A. CRASH SITES ----------------------------------------------------------

class TestGetPersonYearsWithOutcome(unittest.TestCase):
    """get_person_years_with_outcome_by_end_of_wave (person.py:802-810)
    iterates _outcomes[type] WITHOUT filtering priorToSim, then calls
    get_wave_for_age(x[0]). With age=None this is TypeError."""

    def test_does_not_crash_on_priorToSim_only(self):
        # EXPECTED FAILURE today: get_wave_for_age(None) raises TypeError.
        person = _build_person()
        _add_priorToSim_stroke(person)
        result = person.get_person_years_with_outcome_by_end_of_wave(
            outcomeType=OutcomeType.STROKE, wave=0)
        self.assertEqual(0, result)

    def test_counts_only_in_sim_outcomes_when_priorToSim_present(self):
        # EXPECTED FAILURE today: same TypeError as above.
        person = _build_person()
        person._age = [60, 61, 62, 63]
        person._waveCompleted = 2
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=62)
        result = person.get_person_years_with_outcome_by_end_of_wave(
            outcomeType=OutcomeType.STROKE, wave=3)
        self.assertEqual(1, result)

    def test_outcome_in_every_wave_reaches_wave_plus_one(self):
        #waves 0..wave inclusive can contribute wave+1 person-years, the sanity check must allow that
        person = _build_person()
        person._age = [60, 61]
        person._waveCompleted = 1
        _add_in_sim_stroke(person, age=60)
        _add_in_sim_stroke(person, age=61)
        result = person.get_person_years_with_outcome_by_end_of_wave(
            outcomeType=OutcomeType.STROKE, wave=1)
        self.assertEqual(2, result)


# --- B. SILENT LOGIC BUGS ----------------------------------------------------

class TestHasIncidentEvent(unittest.TestCase):
    """has_incident_event (person.py:460-467) inspects _outcomes[type][0][0],
    which is the priorToSim entry's age (None) when one exists. Since
    None == int_age is False, real in-sim incidents at _age[-2] are missed."""

    def test_detects_in_sim_incident_when_priorToSim_is_at_index_0(self):
        # EXPECTED FAILURE today: function returns False because [0][0] is None.
        person = _build_person()
        person._age = [60, 61, 62]
        person._waveCompleted = 1
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=61)  # age == _age[-2]
        self.assertTrue(person.has_incident_event(OutcomeType.STROKE))


class TestGetAgesWithOutcome(unittest.TestCase):
    """get_ages_with_outcome (person.py:822-824) maps every outcome to its age
    without filtering priorToSim, so the returned list contains None when
    priorToSim entries are present."""

    def test_does_not_include_None_for_priorToSim(self):
        # EXPECTED FAILURE today: returned list is [None, 65].
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=65)
        ages = person.get_ages_with_outcome(outcomeType=OutcomeType.STROKE)
        self.assertNotIn(None, ages)
        self.assertEqual([65], ages)


# --- C. CURRENTLY-CORRECT BEHAVIOR (regression guards) -----------------------

class TestHasOutcomeAtAge(unittest.TestCase):
    """has_outcome_at_age (person.py:646-650) checks tuple[0] == age. With
    None != int, priorToSim entries quietly do not match. Pin this in."""

    def test_returns_False_for_priorToSim_only(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertFalse(person.has_outcome_at_age(OutcomeType.STROKE, 60))

    def test_returns_True_for_in_sim_outcome_at_matching_age(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=65)
        self.assertTrue(person.has_outcome_at_age(OutcomeType.STROKE, 65))


class TestHasOutcomeByAge(unittest.TestCase):
    """has_outcome_by_age (person.py:652-656) was fixed (& -> and, reordered)
    so the priorToSim guard short-circuits before the age comparison.
    Lock in that the priorToSim entry is correctly skipped."""

    def test_priorToSim_outcome_is_skipped(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        # No in-sim outcome present; priorToSim must not count regardless of threshold.
        self.assertFalse(person.has_outcome_by_age(OutcomeType.STROKE, 100))

    def test_in_sim_outcome_is_counted_when_under_threshold(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=63)
        self.assertTrue(person.has_outcome_by_age(OutcomeType.STROKE, 65))


class TestGetAgeAtLastOutcome(unittest.TestCase):
    """get_age_at_last_outcome (person.py:687-689) returns the last outcome's
    age unfiltered; for a priorToSim-only person that is None. Callers must
    handle None — pin in the contract."""

    def test_returns_None_when_only_priorToSim(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertIsNone(person.get_age_at_last_outcome(OutcomeType.STROKE))

    def test_returns_real_age_when_in_sim_added_after_priorToSim(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=65)
        self.assertEqual(65, person.get_age_at_last_outcome(OutcomeType.STROKE))


class TestGetAgeAtFirstOutcome(unittest.TestCase):
    """get_age_at_first_outcome (person.py:665-669). Default inSim=True filters
    priorToSim. inSim=False does not — so first age can legitimately be None."""

    def test_inSim_True_returns_None_when_only_priorToSim(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        self.assertIsNone(person.get_age_at_first_outcome(OutcomeType.STROKE, inSim=True))

    def test_inSim_False_returns_None_when_priorToSim_is_first(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=65)
        # With inSim=False the priorToSim entry is included and is at index 0.
        # Documented behavior: caller gets the priorToSim sentinel (None).
        self.assertIsNone(person.get_age_at_first_outcome(OutcomeType.STROKE, inSim=False))

    def test_inSim_True_returns_in_sim_age_when_priorToSim_first(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        _add_in_sim_stroke(person, age=65)
        self.assertEqual(65, person.get_age_at_first_outcome(OutcomeType.STROKE, inSim=True))


class TestGetAgesWithoutOutcome(unittest.TestCase):
    """get_ages_without_outcome (person.py:826-831) does set(_age) - set(ages_with_outcome).
    The None from priorToSim won't match any int age, so the operation is a quiet no-op
    rather than a crash. Pin in correct behavior."""

    def test_baseline_age_remains_when_only_priorToSim_outcome_exists(self):
        person = _build_person()
        _add_priorToSim_stroke(person)
        ages_without = person.get_ages_without_outcome(outcomeType=OutcomeType.STROKE)
        self.assertIn(60, ages_without)


if __name__ == "__main__":
    unittest.main()
