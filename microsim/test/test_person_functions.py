"""Tests for every non-reporting function in person.py, organized by the same
section numbering as the source file. Section 12 (outcome attribute slicing) is
covered separately in test_attr_around_outcomes.py.

Persons are constructed directly (not via PersonFactory) so each test controls
the exact state; simulation-engine tests use small mock repositories/strategies."""

import copy
import math
import unittest
import numpy as np

from microsim.person.person import Person
from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.outcomes.stroke_outcome import StrokeOutcome
from microsim.outcomes.cognition_outcome import (
    CognitionOutcome,
    MMSE_CEILING, MMSE_LOGISTIC_OFFSET, MMSE_LOGISTIC_GCP_SLOPE, MMSE_LOGISTIC_SHAPE,
    GCP_POPULATION_SD, CI_GCP_CHANGE_SD_FACTOR,
    GCP_MEAN_INTERCEPT, GCP_MEAN_AGE_COEFFICIENT, GCP_MEAN_YEARS_IN_SIM_COEFFICIENT,
    GCP_MEAN_SD, MCI_GCP_SD_FACTOR,
)
from microsim.outcomes.qaly_outcome import QALYOutcome
from microsim.outcomes.wmh_outcome import WMHOutcome, scdGroupMap
from microsim.outcomes.wmh_severity import WMHSeverity, wmhSeverityGroupMap
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.modality import Modality, modalityGroupMap
from microsim.risk_factors.a1c import convert_a1c_to_fasting_glucose
from microsim.risk_factors.risk_model_repository import RiskModelRepository
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType, TreatmentStrategyStatus

BP = TreatmentStrategiesType.BP.value


def _build_person(age=60, sbp=120, dbp=80, a1c=5.5, hdl=50, antiHypertensiveCount=0,
                  gender=NHANESGender.MALE, raceEthnicity=RaceEthnicity.NON_HISPANIC_WHITE,
                  smokingStatus=SmokingStatus.NEVER, modality=Modality.NO.value):
    staticRiskFactors = {
        StaticRiskFactorsType.GENDER.value: gender,
        StaticRiskFactorsType.RACE_ETHNICITY.value: raceEthnicity,
        StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE,
        StaticRiskFactorsType.SMOKING_STATUS.value: smokingStatus,
        StaticRiskFactorsType.MODALITY.value: modality}
    dynamicRiskFactors = {
        DynamicRiskFactorsType.AGE.value: age,
        DynamicRiskFactorsType.SBP.value: sbp,
        DynamicRiskFactorsType.DBP.value: dbp,
        DynamicRiskFactorsType.A1C.value: a1c,
        DynamicRiskFactorsType.HDL.value: hdl,
        DynamicRiskFactorsType.LDL.value: 90,
        DynamicRiskFactorsType.TRIG.value: 150,
        DynamicRiskFactorsType.TOT_CHOL.value: 200,
        DynamicRiskFactorsType.BMI.value: 25,
        DynamicRiskFactorsType.WAIST.value: 90,
        DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
        DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE,
        DynamicRiskFactorsType.CREATININE.value: 0.9,
        DynamicRiskFactorsType.AFIB.value: False,
        DynamicRiskFactorsType.PVD.value: False}
    defaultTreatments = {
        DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: antiHypertensiveCount,
        DefaultTreatmentsType.STATIN.value: 0}
    treatmentStrategies = {tst.value: {"status": None} for tst in TreatmentStrategiesType}
    outcomes = {ot: [] for ot in OutcomeType}
    return Person("testPerson", staticRiskFactors, dynamicRiskFactors, defaultTreatments,
                  treatmentStrategies, outcomes)


def _set_waves(person, nWaves):
    """Simulates nWaves completed advances: age list grows to nWaves+1 entries, other
       dynamic risk factors and treatments repeat their baseline value."""
    for rf in person._dynamicRiskFactors:
        values = getattr(person, "_"+rf)
        if rf == DynamicRiskFactorsType.AGE.value:
            setattr(person, "_"+rf, [values[0]+i for i in range(nWaves+1)])
        else:
            setattr(person, "_"+rf, values + [values[-1]]*nWaves)
    for treatment in person._defaultTreatments:
        values = getattr(person, "_"+treatment)
        setattr(person, "_"+treatment, values + [values[-1]]*nWaves)
    person._waveCompleted = nWaves


def _stroke(age, fatal=False):
    return (age, StrokeOutcome(fatal, None, None, None))


def _prior_stroke():
    return (None, StrokeOutcome(False, None, None, None, priorToSim=True))


def _mi(age, fatal=False):
    return (age, Outcome(OutcomeType.MI, fatal))


def _prior_mi():
    return (None, Outcome(OutcomeType.MI, False, priorToSim=True))


def _death(age):
    return (age, Outcome(OutcomeType.DEATH, True))


def _cognition(age, gcp, priorToSim=False):
    return (age, CognitionOutcome(False, priorToSim, gcp))


class _ConstantModel:
    def __init__(self, valueOrCallable):
        self._value = valueOrCallable

    def estimate_next_risk(self, person):
        return self._value(person) if callable(self._value) else self._value


class _ModelRepository(RiskModelRepository):
    """Returns the current last value for every model unless overridden.
       Subclasses RiskModelRepository so the base get_model applies bounds like production."""
    def __init__(self, overrides=None):
        super().__init__()
        self._overrides = overrides if overrides is not None else {}

    def get_model(self, name):
        if name not in self._repository:
            if name in self._overrides:
                self._repository[name] = _ConstantModel(self._overrides[name])
            else:
                self._repository[name] = _ConstantModel(lambda person, name=name: getattr(person, "_"+name)[-1])
        return super().get_model(name)


class _PerTypeOutcomeRepository:
    def __init__(self, outcomeFactory=None):
        self._outcomeFactory = outcomeFactory

    def select_outcome_model_for_person(self, person):
        return self

    def get_next_outcome(self, person):
        return self._outcomeFactory(person) if self._outcomeFactory is not None else None

    def get_prevalent_outcome(self, person):
        return self._outcomeFactory(person) if self._outcomeFactory is not None else None


class _OutcomeRepository:
    def __init__(self, factories=None):
        factories = factories if factories is not None else {}
        self._repository = {ot: _PerTypeOutcomeRepository(factories.get(ot)) for ot in OutcomeType}


class _NoneOutcomeRepository:
    """Prevalence-style repository where unregistered outcome types are None."""
    def __init__(self, factories):
        self._repository = {ot: None for ot in OutcomeType}
        for ot, factory in factories.items():
            self._repository[ot] = _PerTypeOutcomeRepository(factory)


class _Strategy:
    def __init__(self, status, updatedTreatments=None, updatedRiskFactors=None):
        self.status = status
        self._updatedTreatments = updatedTreatments if updatedTreatments is not None else {}
        self._updatedRiskFactors = updatedRiskFactors if updatedRiskFactors is not None else {}

    def get_updated_treatments(self, person):
        return dict(self._updatedTreatments)

    def get_updated_risk_factors(self, person):
        return dict(self._updatedRiskFactors)


class _StrategyRepository:
    def __init__(self, bpStrategy=None):
        self._repository = {tst.value: None for tst in TreatmentStrategiesType}
        self._repository[BP] = bpStrategy


# ==========================================================================
# 1. Construction & identity
# ==========================================================================

class TestInit(unittest.TestCase):

    def test_initial_state(self):
        person = _build_person()
        self.assertEqual("testPerson", person._name)
        self.assertIsNone(person._index)
        self.assertEqual(-1, person._waveCompleted)
        self.assertEqual({}, person._randomEffects)
        self.assertEqual(NHANESGender.MALE, person._gender)  # static: scalar
        self.assertEqual([60], person._age)  # dynamic: single-element list
        self.assertEqual([0], person._antiHypertensiveCount)  # treatment: single-element list
        self.assertEqual([], person._outcomes[OutcomeType.STROKE])
        self.assertIsNone(person._treatmentStrategies[BP]["status"])


# ==========================================================================
# 2. Simulation engine
# ==========================================================================

class TestAdvance(unittest.TestCase):

    def test_first_advance_skips_risk_factor_and_treatment_advance(self):
        person = _build_person()
        person.advance(1, _ModelRepository(), _ModelRepository(), _OutcomeRepository())
        self.assertEqual(0, person._waveCompleted)
        self.assertEqual(1, len(person._age))

    def test_second_advance_appends_risk_factors_and_treatments(self):
        person = _build_person()
        person.advance(2, _ModelRepository({"age": lambda p: p._age[-1]+1}), _ModelRepository(), _OutcomeRepository())
        self.assertEqual(1, person._waveCompleted)
        self.assertEqual([60, 61], person._age)
        self.assertEqual(2, len(person._sbp))
        self.assertEqual(2, len(person._statin))

    def test_advance_stops_after_death(self):
        person = _build_person()
        deathRepo = _OutcomeRepository({OutcomeType.DEATH: lambda p: Outcome(OutcomeType.DEATH, True)})
        person.advance(3, _ModelRepository(), _ModelRepository(), deathRepo)
        self.assertEqual(0, person._waveCompleted)  # died in the first wave, no further advances
        self.assertEqual(1, len(person._age))

    def test_dead_person_does_not_advance(self):
        person = _build_person()
        person._outcomes[OutcomeType.DEATH].append(_death(60))
        person.advance(1, _ModelRepository(), _ModelRepository(), _OutcomeRepository())
        self.assertEqual(-1, person._waveCompleted)


class TestAdvanceRiskFactorsAndTreatments(unittest.TestCase):

    def test_advance_risk_factors_appends_model_estimate(self):
        person = _build_person()
        person.advance_risk_factors(_ModelRepository({"sbp": 130}))
        self.assertEqual([120, 130], person._sbp)

    def test_advance_risk_factors_applies_bounds(self):
        person = _build_person()
        person.advance_risk_factors(_ModelRepository({"sbp": 500}))
        self.assertEqual(297., person._sbp[-1])

    def test_advance_risk_factors_applies_child_bounds(self):
        person = _build_person(age=10)
        person.advance_risk_factors(_ModelRepository({"age": lambda p: p._age[-1]+1, "sbp": 500}))
        self.assertEqual(11, person._age[-1])
        self.assertEqual(190.3, person._sbp[-1])

    def test_advance_risk_factors_crosses_into_adulthood(self):
        person = _build_person(age=17)
        person.advance_risk_factors(_ModelRepository({"age": lambda p: p._age[-1]+1, "sbp": 500}))
        self.assertEqual(18, person._age[-1])
        self.assertEqual(297., person._sbp[-1])

    def test_get_next_risk_factor(self):
        person = _build_person()
        self.assertEqual(130, person.get_next_risk_factor("sbp", _ModelRepository({"sbp": 130})))

    def test_advance_treatments_appends_model_estimate(self):
        person = _build_person()
        person.advance_treatments(_ModelRepository({"statin": 1}))
        self.assertEqual([0, 1], person._statin)

    def test_get_next_treatment(self):
        person = _build_person()
        self.assertEqual(2, person.get_next_treatment("statin", _ModelRepository({"statin": 2})))


class TestAdvanceTreatmentStrategies(unittest.TestCase):

    def test_no_strategies_keeps_status_none(self):
        person = _build_person()
        person.advance_treatment_strategies_and_update_risk_factors(None)
        for tst in TreatmentStrategiesType:
            self.assertIsNone(person._treatmentStrategies[tst.value]["status"])

    def test_strategy_updates_status_treatments_and_risk_factors(self):
        person = _build_person()
        strategy = _Strategy(TreatmentStrategyStatus.BEGIN,
                             updatedTreatments={"antiHypertensiveCount": 2},
                             updatedRiskFactors={"sbp": 110})
        person.advance_treatment_strategies_and_update_risk_factors(_StrategyRepository(strategy))
        self.assertEqual(TreatmentStrategyStatus.BEGIN, person._treatmentStrategies[BP]["status"])
        self.assertEqual([2], person._antiHypertensiveCount)  # updated in place, not appended
        self.assertEqual([110], person._sbp)

    def test_update_treatments_only_touches_listed_treatments(self):
        person = _build_person()
        person.update_treatments(_Strategy(None, updatedTreatments={"statin": 1}))
        self.assertEqual([1], person._statin)
        self.assertEqual([0], person._antiHypertensiveCount)

    def test_update_risk_factors_applies_bounds(self):
        person = _build_person()
        person.update_risk_factors(_Strategy(None, updatedRiskFactors={"sbp": 20}))
        self.assertEqual(58.20, person._sbp[-1])


class TestUpdateTreatmentStrategyStatus(unittest.TestCase):

    def _person_with_status(self, status):
        person = _build_person()
        person._treatmentStrategies[BP]["status"] = status
        return person

    def _assert_transition(self, personStatus, strategyStatus, expected):
        person = self._person_with_status(personStatus)
        strategy = _Strategy(strategyStatus) if strategyStatus is not None else None
        person.update_treatment_strategy_status(strategy, TreatmentStrategiesType.BP)
        self.assertEqual(expected, person._treatmentStrategies[BP]["status"])

    def _assert_invalid(self, personStatus, strategyStatus):
        person = self._person_with_status(personStatus)
        strategy = _Strategy(strategyStatus) if strategyStatus is not None else None
        with self.assertRaises(RuntimeError):
            person.update_treatment_strategy_status(strategy, TreatmentStrategiesType.BP)

    def test_valid_transitions(self):
        self._assert_transition(None, TreatmentStrategyStatus.BEGIN, TreatmentStrategyStatus.BEGIN)
        self._assert_transition(TreatmentStrategyStatus.BEGIN, TreatmentStrategyStatus.MAINTAIN, TreatmentStrategyStatus.MAINTAIN)
        self._assert_transition(TreatmentStrategyStatus.BEGIN, TreatmentStrategyStatus.END, TreatmentStrategyStatus.END)
        self._assert_transition(TreatmentStrategyStatus.MAINTAIN, TreatmentStrategyStatus.MAINTAIN, TreatmentStrategyStatus.MAINTAIN)
        self._assert_transition(TreatmentStrategyStatus.MAINTAIN, TreatmentStrategyStatus.END, TreatmentStrategyStatus.END)
        self._assert_transition(TreatmentStrategyStatus.END, TreatmentStrategyStatus.BEGIN, TreatmentStrategyStatus.BEGIN)
        self._assert_transition(None, None, None)
        self._assert_transition(TreatmentStrategyStatus.END, None, None)

    def test_invalid_transitions_raise(self):
        self._assert_invalid(None, TreatmentStrategyStatus.MAINTAIN)
        self._assert_invalid(TreatmentStrategyStatus.BEGIN, TreatmentStrategyStatus.BEGIN)
        self._assert_invalid(TreatmentStrategyStatus.MAINTAIN, TreatmentStrategyStatus.BEGIN)
        self._assert_invalid(TreatmentStrategyStatus.END, TreatmentStrategyStatus.END)
        self._assert_invalid(TreatmentStrategyStatus.BEGIN, None)
        self._assert_invalid(TreatmentStrategyStatus.MAINTAIN, None)


class TestAdvanceOutcomes(unittest.TestCase):

    def test_outcome_added_at_current_age(self):
        person = _build_person()
        repo = _OutcomeRepository({OutcomeType.STROKE: lambda p: StrokeOutcome(False, None, None, None)})
        person.advance_outcomes(repo)
        self.assertEqual(1, len(person._outcomes[OutcomeType.STROKE]))
        self.assertEqual(60, person._outcomes[OutcomeType.STROKE][0][0])

    def test_none_outcomes_not_added(self):
        person = _build_person()
        person.advance_outcomes(_OutcomeRepository())
        for ot in OutcomeType:
            self.assertEqual([], person._outcomes[ot])

    def test_seed_prevalent_outcomes_skips_none_and_adds_priorToSim(self):
        person = _build_person()
        repo = _NoneOutcomeRepository({OutcomeType.STROKE: lambda p: StrokeOutcome(False, None, None, None, priorToSim=True)})
        person.seed_prevalent_outcomes(repo)
        self.assertEqual(1, len(person._outcomes[OutcomeType.STROKE]))
        self.assertIsNone(person._outcomes[OutcomeType.STROKE][0][0])  # priorToSim carries age=None
        self.assertEqual([], person._outcomes[OutcomeType.MI])

    def test_get_outcomes_in_order_follows_enum_order(self):
        person = _build_person()
        self.assertEqual(list(OutcomeType), person.get_outcomes_in_order())

    def test_get_outcomes_in_order_with_subset(self):
        person = _build_person()
        person._outcomes = {OutcomeType.DEATH: [], OutcomeType.STROKE: []}
        inOrder = person.get_outcomes_in_order()
        self.assertEqual([OutcomeType.STROKE, OutcomeType.DEATH], inOrder)

    def test_add_outcome(self):
        person = _build_person()
        person.add_outcome(None)
        self.assertEqual([], person._outcomes[OutcomeType.STROKE])
        person.add_outcome(StrokeOutcome(False, None, None, None))
        self.assertEqual((60, person._outcomes[OutcomeType.STROKE][0][1]), person._outcomes[OutcomeType.STROKE][0])
        person.add_outcome(StrokeOutcome(False, None, None, None, priorToSim=True))
        self.assertIsNone(person._outcomes[OutcomeType.STROKE][1][0])


# ==========================================================================
# 3. Treatment-strategy state
# ==========================================================================

class TestTreatmentStrategyState(unittest.TestCase):

    def _person_with_bp_status(self, status, medsAdded=None):
        person = _build_person()
        person._treatmentStrategies[BP]["status"] = status
        if medsAdded is not None:
            person._treatmentStrategies[BP]["bpMedsAdded"] = medsAdded
        return person

    def test_is_in_treatment_strategy_by_status(self):
        self.assertFalse(self._person_with_bp_status(None).is_in_treatment_strategy(BP))
        self.assertTrue(self._person_with_bp_status(TreatmentStrategyStatus.BEGIN).is_in_treatment_strategy(BP))
        self.assertTrue(self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN).is_in_treatment_strategy(BP))
        self.assertFalse(self._person_with_bp_status(TreatmentStrategyStatus.END).is_in_treatment_strategy(BP))

    def test_is_in_bp_treatment(self):
        self.assertTrue(self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN).is_in_bp_treatment)
        self.assertFalse(self._person_with_bp_status(None).is_in_bp_treatment)

    def test_is_in_any_treatment_strategy(self):
        self.assertFalse(_build_person().is_in_any_treatment_strategy())
        self.assertTrue(self._person_with_bp_status(TreatmentStrategyStatus.BEGIN).is_in_any_treatment_strategy())

    def test_get_treatment_strategies_with_participation(self):
        self.assertEqual([], _build_person().get_treatment_strategies_with_participation())
        person = self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN)
        self.assertEqual([BP], person.get_treatment_strategies_with_participation())

    def test_has_meds_added(self):
        self.assertIsNone(_build_person().has_meds_added(BP))  # not in strategy
        self.assertFalse(self._person_with_bp_status(TreatmentStrategyStatus.BEGIN).has_meds_added(BP))  # key not set yet
        self.assertFalse(self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN, medsAdded=0).has_meds_added(BP))
        self.assertTrue(self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN, medsAdded=2).has_meds_added(BP))

    def test_has_any_meds_added(self):
        self.assertIsNone(_build_person().has_any_meds_added())
        self.assertFalse(self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN, medsAdded=0).has_any_meds_added())
        self.assertTrue(self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN, medsAdded=1).has_any_meds_added())

    def test_get_treatment_strategies_with_meds_added(self):
        self.assertEqual([], _build_person().get_treatment_strategies_with_meds_added())
        person = self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN, medsAdded=3)
        self.assertEqual([BP], person.get_treatment_strategies_with_meds_added())

    def test_get_meds_added(self):
        self.assertEqual(0, _build_person().get_meds_added(BP))  # key absent defaults to 0
        self.assertEqual(3, self._person_with_bp_status(TreatmentStrategyStatus.MAINTAIN, medsAdded=3).get_meds_added(BP))

    def test_antiHypertensiveCountPlusBPMedsAdded(self):
        person = _build_person(antiHypertensiveCount=1)
        self.assertEqual(1, person._antiHypertensiveCountPlusBPMedsAdded())  # not in strategy
        person._treatmentStrategies[BP]["status"] = TreatmentStrategyStatus.BEGIN
        self.assertEqual(1, person._antiHypertensiveCountPlusBPMedsAdded())  # meds not counted at BEGIN
        person._treatmentStrategies[BP]["status"] = TreatmentStrategyStatus.MAINTAIN
        person._treatmentStrategies[BP]["bpMedsAdded"] = 2
        self.assertEqual(3, person._antiHypertensiveCountPlusBPMedsAdded())

    def test_any_antiHypertensive(self):
        self.assertFalse(_build_person(antiHypertensiveCount=0)._any_antiHypertensive)
        self.assertTrue(_build_person(antiHypertensiveCount=1)._any_antiHypertensive)

    def test_get_last_default_treatment(self):
        person = _build_person(antiHypertensiveCount=2)
        self.assertEqual(2, person.get_last_default_treatment(DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value))


# ==========================================================================
# 4. Time / age / wave
# ==========================================================================

class TestTimeAgeWave(unittest.TestCase):

    def test_current_age(self):
        person = _build_person()
        _set_waves(person, 2)
        self.assertEqual(62, person._current_age)

    def test_get_wave_for_age(self):
        person = _build_person()
        _set_waves(person, 2)
        self.assertEqual(0, person.get_wave_for_age(60))
        self.assertEqual(2, person.get_wave_for_age(62))
        with self.assertRaises(RuntimeError):
            person.get_wave_for_age(59)
        with self.assertRaises(RuntimeError):
            person.get_wave_for_age(63)

    def test_get_age_for_wave(self):
        person = _build_person()
        _set_waves(person, 2)
        self.assertEqual(61, person.get_age_for_wave(1))
        with self.assertRaises(RuntimeError):
            person.get_age_for_wave(3)

    def test_valid_wave(self):
        person = _build_person()
        _set_waves(person, 2)
        self.assertFalse(person.valid_wave(-1))
        self.assertTrue(person.valid_wave(0))
        self.assertTrue(person.valid_wave(2))
        self.assertFalse(person.valid_wave(3))

    def test_get_ages(self):
        person = _build_person()
        _set_waves(person, 2)
        self.assertEqual([60, 61, 62], person.get_ages())

    def test_get_median_age(self):
        person = _build_person()
        _set_waves(person, 4)  # ages 60..64
        self.assertEqual(62, person.get_median_age())

    def test_get_years_in_simulation(self):
        person = _build_person()
        self.assertEqual(0, person.get_years_in_simulation())
        _set_waves(person, 3)
        self.assertEqual(4, person.get_years_in_simulation())

    def test_get_gender_age_of_all_outcomes_in_sim(self):
        person = _build_person()
        _set_waves(person, 2)
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        genderValue = person._gender.value
        self.assertEqual([(genderValue, 61)], person.get_gender_age_of_all_outcomes_in_sim(OutcomeType.STROKE))
        self.assertEqual([], person.get_gender_age_of_all_outcomes_in_sim(OutcomeType.MI))

    def test_get_gender_age_of_all_years_in_sim(self):
        person = _build_person()
        _set_waves(person, 1)
        genderValue = person._gender.value
        self.assertEqual([(genderValue, 60), (genderValue, 61)], person.get_gender_age_of_all_years_in_sim())


# ==========================================================================
# 5. Life and death
# ==========================================================================

class TestLifeAndDeath(unittest.TestCase):

    def test_is_alive_and_is_dead(self):
        person = _build_person()
        self.assertTrue(person.is_alive)
        self.assertFalse(person.is_dead)
        person._outcomes[OutcomeType.DEATH].append(_death(60))
        self.assertFalse(person.is_alive)
        self.assertTrue(person.is_dead)

    def test_get_death_age(self):
        person = _build_person()
        self.assertIsNone(person.get_death_age())
        person._outcomes[OutcomeType.DEATH].append(_death(60))
        self.assertEqual(60, person.get_death_age())

    def test_is_alive_at_index_without_death(self):
        person = _build_person()
        _set_waves(person, 2)
        self.assertTrue(person.is_alive_at_index(0))
        self.assertTrue(person.is_alive_at_index(-1))

    def test_is_alive_at_index_with_death(self):
        person = _build_person()
        _set_waves(person, 2)
        person._outcomes[OutcomeType.DEATH].append(_death(62))
        self.assertTrue(person.is_alive_at_index(0))
        self.assertTrue(person.is_alive_at_index(1))
        self.assertFalse(person.is_alive_at_index(2))
        self.assertFalse(person.is_alive_at_index(-1))

    def test_is_alive_at_index_rejects_negative_indices_other_than_minus_one(self):
        person = _build_person()
        _set_waves(person, 2)
        with self.assertRaises(RuntimeError):
            person.is_alive_at_index(-2)

    def test_minus_one_agrees_with_last_wave_for_alive_person(self):
        person = _build_person()
        _set_waves(person, 2)
        self.assertEqual(person.is_alive_at_index(person._waveCompleted), person.is_alive_at_index(-1))
        self.assertTrue(person.is_alive_at_index(-1))

    def test_minus_one_agrees_with_death_wave_for_dead_person(self):
        person = _build_person()
        _set_waves(person, 2)
        person._outcomes[OutcomeType.DEATH].append(_death(62))  # died during wave 2, _waveCompleted stays 2
        self.assertEqual(person.is_alive_at_index(person._waveCompleted), person.is_alive_at_index(-1))
        self.assertFalse(person.is_alive_at_index(-1))


# ==========================================================================
# 6. Outcomes - generic queries
# ==========================================================================

class TestGenericOutcomeQueries(unittest.TestCase):

    def _person_with_stroke_at_61(self, nWaves=2):
        person = _build_person()
        _set_waves(person, nWaves)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        return person

    def test_has_outcome_at_current_age(self):
        person = _build_person()
        _set_waves(person, 1)
        self.assertFalse(person.has_outcome_at_current_age(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertTrue(person.has_outcome_at_current_age(OutcomeType.STROKE))

    def test_has_outcome_at_current_age_priorToSim_only(self):
        person = _build_person()
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        self.assertFalse(person.has_outcome_at_current_age(OutcomeType.STROKE))

    def test_has_fatal_outcome_at_current_age(self):
        person = _build_person()
        _set_waves(person, 1)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61, fatal=True))
        self.assertTrue(person.has_fatal_outcome_at_current_age(OutcomeType.STROKE))
        person._outcomes[OutcomeType.MI].append(_mi(61, fatal=False))
        self.assertFalse(person.has_fatal_outcome_at_current_age(OutcomeType.MI))
        self.assertFalse(person.has_fatal_outcome_at_current_age(OutcomeType.DEMENTIA))

    def test_has_outcome_prior_to_and_during_simulation(self):
        person = _build_person()
        self.assertFalse(person.has_outcome_prior_to_simulation(OutcomeType.STROKE))
        self.assertFalse(person.has_outcome_during_simulation(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        self.assertTrue(person.has_outcome_prior_to_simulation(OutcomeType.STROKE))
        self.assertFalse(person.has_outcome_during_simulation(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].append(_stroke(60))
        self.assertTrue(person.has_outcome_during_simulation(OutcomeType.STROKE))

    def test_get_outcomes_during_simulation_and_get_outcomes(self):
        person = _build_person()
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        person._outcomes[OutcomeType.STROKE].append(_stroke(60))
        self.assertEqual(1, len(person.get_outcomes_during_simulation(OutcomeType.STROKE)))
        self.assertEqual(1, len(person.get_outcomes(OutcomeType.STROKE, inSim=True)))
        self.assertEqual(2, len(person.get_outcomes(OutcomeType.STROKE, inSim=False)))

    def test_has_outcome_during_simulation_prior_to_wave(self):
        person = self._person_with_stroke_at_61()
        self.assertFalse(person.has_outcome_during_simulation_prior_to_wave(OutcomeType.STROKE, 1))  # strict <
        self.assertTrue(person.has_outcome_during_simulation_prior_to_wave(OutcomeType.STROKE, 2))
        self.assertFalse(person.has_outcome_during_simulation_prior_to_wave(OutcomeType.MI, 2))

    def test_has_outcome(self):
        person = _build_person()
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        self.assertFalse(person.has_outcome(OutcomeType.STROKE, inSim=True))
        self.assertTrue(person.has_outcome(OutcomeType.STROKE, inSim=False))

    def test_has_any_and_all_outcomes(self):
        person = self._person_with_stroke_at_61()
        self.assertTrue(person.has_any_outcome([OutcomeType.STROKE, OutcomeType.MI]))
        self.assertFalse(person.has_all_outcomes([OutcomeType.STROKE, OutcomeType.MI]))
        person._outcomes[OutcomeType.MI].append(_mi(62))
        self.assertTrue(person.has_all_outcomes([OutcomeType.STROKE, OutcomeType.MI]))

    def test_has_outcome_at_age(self):
        person = self._person_with_stroke_at_61()
        self.assertTrue(person.has_outcome_at_age(OutcomeType.STROKE, 61))
        self.assertFalse(person.has_outcome_at_age(OutcomeType.STROKE, 62))

    def test_has_outcome_by_age(self):
        person = self._person_with_stroke_at_61()
        self.assertFalse(person.has_outcome_by_age(OutcomeType.STROKE, 60))
        self.assertTrue(person.has_outcome_by_age(OutcomeType.STROKE, 61))
        self.assertTrue(person.has_outcome_by_age(OutcomeType.STROKE, 62))

    def test_has_outcome_by_age_priorToSim(self):
        person = _build_person()
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        self.assertFalse(person.has_outcome_by_age(OutcomeType.STROKE, 60, inSim=True))
        self.assertTrue(person.has_outcome_by_age(OutcomeType.STROKE, 60, inSim=False))

    def test_has_any_outcome_by_end_of_wave(self):
        person = self._person_with_stroke_at_61()
        self.assertFalse(person.has_any_outcome_by_end_of_wave([OutcomeType.STROKE], wave=0))
        self.assertTrue(person.has_any_outcome_by_end_of_wave([OutcomeType.STROKE], wave=1))
        self.assertFalse(_build_person().has_any_outcome_by_end_of_wave([OutcomeType.STROKE], wave=1))

    def test_has_outcome_during_wave(self):
        person = self._person_with_stroke_at_61()
        self.assertFalse(person.has_outcome_during_wave(0, OutcomeType.STROKE))
        self.assertTrue(person.has_outcome_during_wave(1, OutcomeType.STROKE))
        with self.assertRaises(RuntimeError):
            person.has_outcome_during_wave(3, OutcomeType.STROKE)

    def test_has_outcome_during_or_prior_to_wave_mid_advance(self):
        #mid-advance state: age for the current wave appended, wave not yet completed
        person = _build_person()
        _set_waves(person, 2)
        person._waveCompleted = 1
        person._outcomes[OutcomeType.STROKE].append(_stroke(62))
        self.assertTrue(person.has_outcome_during_or_prior_to_wave(2, OutcomeType.STROKE))
        self.assertFalse(person.has_outcome_during_or_prior_to_wave(1, OutcomeType.STROKE))
        with self.assertRaises(RuntimeError):
            person.has_outcome_during_or_prior_to_wave(3, OutcomeType.STROKE)

    def test_has_incident_event(self):
        person = _build_person()
        _set_waves(person, 2)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))  # first in-sim outcome at _age[-2]
        self.assertTrue(person.has_incident_event(OutcomeType.STROKE))
        person._outcomes[OutcomeType.MI].append(_mi(62))  # at _age[-1], not incident
        self.assertFalse(person.has_incident_event(OutcomeType.MI))

    def test_get_age_at_first_outcome(self):
        person = _build_person()
        self.assertIsNone(person.get_age_at_first_outcome(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        _set_waves(person, 2)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual(61, person.get_age_at_first_outcome(OutcomeType.STROKE, inSim=True))
        self.assertIsNone(person.get_age_at_first_outcome(OutcomeType.STROKE, inSim=False))  # first is priorToSim, age None

    def test_get_first_incidence_age(self):
        person = _build_person()
        self.assertIsNone(person.get_first_incidence_age(OutcomeType.STROKE))
        _set_waves(person, 2)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual(61, person.get_first_incidence_age(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].insert(0, _prior_stroke())
        self.assertIsNone(person.get_first_incidence_age(OutcomeType.STROKE))  # not at risk

    def test_get_min_age_and_wave_of_first_outcomes(self):
        person = _build_person()
        _set_waves(person, 3)
        person._outcomes[OutcomeType.STROKE].append(_stroke(62))
        person._outcomes[OutcomeType.MI].append(_mi(61))
        outcomeList = [OutcomeType.STROKE, OutcomeType.MI]
        self.assertEqual(61, person.get_min_age_of_first_outcomes(outcomeList))
        self.assertEqual(1, person.get_min_wave_of_first_outcomes(outcomeList))
        self.assertIsNone(person.get_min_age_of_first_outcomes([OutcomeType.DEMENTIA]))
        self.assertIsNone(person.get_min_wave_of_first_outcomes([OutcomeType.DEMENTIA]))

    def test_get_min_age_and_wave_of_first_outcomes_or_last(self):
        person = _build_person()
        _set_waves(person, 3)
        self.assertEqual(63, person.get_min_age_of_first_outcomes_or_last_age([OutcomeType.STROKE]))
        self.assertEqual(3, person.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.STROKE]))
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual(61, person.get_min_age_of_first_outcomes_or_last_age([OutcomeType.STROKE]))
        self.assertEqual(1, person.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.STROKE]))

    def test_get_age_at_last_outcome(self):
        person = _build_person()
        self.assertIsNone(person.get_age_at_last_outcome(OutcomeType.STROKE))
        _set_waves(person, 2)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        person._outcomes[OutcomeType.STROKE].append(_stroke(62))
        self.assertEqual(62, person.get_age_at_last_outcome(OutcomeType.STROKE))

    def test_get_age_at_last_outcome_in_sim(self):
        person = _build_person()
        self.assertIsNone(person.get_age_at_last_outcome_in_sim(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        self.assertIsNone(person.get_age_at_last_outcome_in_sim(OutcomeType.STROKE))  # last is priorToSim
        _set_waves(person, 1)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual(61, person.get_age_at_last_outcome_in_sim(OutcomeType.STROKE))


# ==========================================================================
# 7. Outcomes - phenotype item extraction
# ==========================================================================

class TestOutcomeItemExtraction(unittest.TestCase):

    def _person_with_gcps(self):
        person = _build_person()
        _set_waves(person, 1)
        person._outcomes[OutcomeType.COGNITION].append(_cognition(None, 40, priorToSim=True))
        person._outcomes[OutcomeType.COGNITION].append(_cognition(60, 55))
        person._outcomes[OutcomeType.COGNITION].append(_cognition(61, 50))
        return person

    def test_get_outcome_item(self):
        person = self._person_with_gcps()
        self.assertEqual([55, 50], person.get_outcome_item(OutcomeType.COGNITION, "gcp"))
        self.assertEqual([40, 55, 50], person.get_outcome_item(OutcomeType.COGNITION, "gcp", inSim=False))

    def test_get_outcome_item_first_last(self):
        person = self._person_with_gcps()
        self.assertEqual(55, person.get_outcome_item_first(OutcomeType.COGNITION, "gcp"))
        self.assertEqual(50, person.get_outcome_item_last(OutcomeType.COGNITION, "gcp"))

    def test_get_outcome_item_sum_mean_change(self):
        person = self._person_with_gcps()
        self.assertEqual(105, person.get_outcome_item_sum(OutcomeType.COGNITION, "gcp"))
        self.assertAlmostEqual(52.5, person.get_outcome_item_mean(OutcomeType.COGNITION, "gcp"))
        self.assertEqual(-5, person.get_outcome_item_overall_change(OutcomeType.COGNITION, "gcp"))


# ==========================================================================
# 8. Outcome shortcuts
# ==========================================================================

class TestOutcomeShortcuts(unittest.TestCase):

    def test_mi_stroke_dementia_properties(self):
        person = _build_person()
        self.assertFalse(person._mi)
        self.assertFalse(person._stroke)
        self.assertFalse(person._dementia)
        person._outcomes[OutcomeType.MI].append(_mi(60))
        person._outcomes[OutcomeType.STROKE].append(_stroke(60))
        person._outcomes[OutcomeType.DEMENTIA].append((60, Outcome(OutcomeType.DEMENTIA, False)))
        self.assertTrue(person._mi)
        self.assertTrue(person._stroke)
        self.assertTrue(person._dementia)

    def test_selfReport_priorToSim_properties(self):
        person = _build_person()
        self.assertFalse(person._selfReportStrokePriorToSim)
        self.assertFalse(person._selfReportMIPriorToSim)
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        person._outcomes[OutcomeType.MI].append(_mi(60))
        self.assertTrue(person._selfReportStrokePriorToSim)
        self.assertFalse(person._selfReportMIPriorToSim)

    def test_stroke_shortcuts(self):
        person = _build_person()
        _set_waves(person, 1)
        person._outcomes[OutcomeType.STROKE].append(_prior_stroke())
        person._outcomes[OutcomeType.STROKE].append(_stroke(61, fatal=True))
        self.assertTrue(person.has_stroke_prior_to_simulation())
        self.assertTrue(person.has_stroke_during_simulation())
        self.assertTrue(person.has_stroke_during_wave(1))
        self.assertFalse(person.has_stroke_during_wave(0))
        self.assertTrue(person.has_fatal_stroke())

    def test_mi_shortcuts(self):
        person = _build_person()
        _set_waves(person, 1)
        person._outcomes[OutcomeType.MI].append(_mi(61))
        self.assertFalse(person.has_mi_prior_to_simulation())
        self.assertTrue(person.has_mi_during_simulation())
        self.assertTrue(person.has_mi_during_wave(1))
        self.assertFalse(person.has_fatal_mi())

    def test_has_incident_dementia(self):
        person = _build_person()
        _set_waves(person, 2)
        person._outcomes[OutcomeType.DEMENTIA].append((61, Outcome(OutcomeType.DEMENTIA, False)))
        self.assertTrue(person.has_incident_dementia())

    def test_has_epilepsy(self):
        person = _build_person()
        self.assertFalse(person.has_epilepsy())
        person._outcomes[OutcomeType.EPILEPSY].append((None, Outcome(OutcomeType.EPILEPSY, False, priorToSim=True)))
        self.assertTrue(person.has_epilepsy())

    def test_has_diabetes_is_sticky(self):
        person = _build_person(a1c=5.5)
        self.assertFalse(person.has_diabetes())
        self.assertFalse(person._diabetes)
        person._a1c = [5.5, 7.0, 5.8]  # crossed 6.5 in the past, dropped since
        self.assertTrue(person.has_diabetes())
        self.assertTrue(person._diabetes)

    def test_gfr_and_current_ckd(self):
        person = _build_person()
        self.assertEqual(person._current_ckd, person._gfr < 60)
        self.assertFalse(person._current_ckd)  # healthy creatinine
        person._creatinine = [8.0]
        self.assertTrue(person._current_ckd)


# ==========================================================================
# 9. Risk-factor derived shortcuts
# ==========================================================================

class TestRiskFactorShortcuts(unittest.TestCase):

    def test_current_smoker(self):
        self.assertFalse(_build_person(smokingStatus=SmokingStatus.NEVER)._current_smoker)
        self.assertTrue(_build_person(smokingStatus=SmokingStatus.CURRENT)._current_smoker)

    def test_black_and_white(self):
        black = _build_person(raceEthnicity=RaceEthnicity.NON_HISPANIC_BLACK)
        white = _build_person(raceEthnicity=RaceEthnicity.NON_HISPANIC_WHITE)
        self.assertTrue(black._black)
        self.assertFalse(black._white)
        self.assertTrue(white._white)
        self.assertFalse(white._black)


# ==========================================================================
# 10. Cognition
# ==========================================================================

class TestCognition(unittest.TestCase):

    def test_baselineGcp(self):
        person = _build_person()
        with self.assertRaises(RuntimeError):
            person._baselineGcp
        person._outcomes[OutcomeType.COGNITION].append(_cognition(None, 40, priorToSim=True))
        with self.assertRaises(RuntimeError):
            person._baselineGcp  # priorToSim cognition is not a baseline
        person._outcomes[OutcomeType.COGNITION].append(_cognition(60, 55))
        person._outcomes[OutcomeType.COGNITION].append(_cognition(61, 50))
        self.assertEqual(55, person._baselineGcp)

    def test_gcpSlope(self):
        person = _build_person()
        self.assertEqual(0, person._gcpSlope)
        person._outcomes[OutcomeType.COGNITION].append(_cognition(60, 55))
        self.assertEqual(0, person._gcpSlope)
        person._outcomes[OutcomeType.COGNITION].append(_cognition(61, 52))
        self.assertEqual(-3, person._gcpSlope)

    def test_get_current_mmse(self):
        person = _build_person()
        gcp = 55
        person._outcomes[OutcomeType.COGNITION].append(_cognition(60, gcp))
        expected = MMSE_CEILING / ((MMSE_LOGISTIC_OFFSET + np.exp(-MMSE_LOGISTIC_GCP_SLOPE * gcp)) ** (1 / MMSE_LOGISTIC_SHAPE))
        self.assertAlmostEqual(expected, person.get_current_mmse())

    def test_has_cognitive_impairment(self):
        threshold = CI_GCP_CHANGE_SD_FACTOR * GCP_POPULATION_SD
        person = _build_person()
        person._outcomes[OutcomeType.COGNITION].append(_cognition(60, 55))
        person._outcomes[OutcomeType.COGNITION].append(_cognition(61, 55 - threshold - 1))
        self.assertTrue(person.has_cognitive_impairment())
        self.assertTrue(person.has_ci())
        person2 = _build_person()
        person2._outcomes[OutcomeType.COGNITION].append(_cognition(60, 55))
        person2._outcomes[OutcomeType.COGNITION].append(_cognition(61, 55 - threshold + 1))
        self.assertFalse(person2.has_cognitive_impairment())

    def test_has_mild_cognitive_impairment(self):
        person = _build_person()
        gcpMean = (GCP_MEAN_INTERCEPT + GCP_MEAN_AGE_COEFFICIENT * person._current_age
                   + GCP_MEAN_YEARS_IN_SIM_COEFFICIENT * person.get_years_in_simulation())
        gcpCutoff = gcpMean - MCI_GCP_SD_FACTOR * GCP_MEAN_SD
        person._outcomes[OutcomeType.COGNITION].append(_cognition(60, gcpCutoff - 1))
        self.assertTrue(person.has_mild_cognitive_impairment())
        self.assertTrue(person.has_mci())
        person2 = _build_person()
        person2._outcomes[OutcomeType.COGNITION].append(_cognition(60, gcpCutoff + 1))
        self.assertFalse(person2.has_mild_cognitive_impairment())


# ==========================================================================
# 11. WMH / SCD classification
# ==========================================================================

class TestWMHClassification(unittest.TestCase):

    def _person_with_wmh(self, sbi=False, wmh=True, severityUnknown=False,
                         severity=WMHSeverity.MILD, modality=Modality.MR.value):
        person = _build_person(modality=modality)
        person._outcomes[OutcomeType.WMH].append(
            (60, WMHOutcome(False, sbi, wmh, severityUnknown, severity)))
        return person

    def test_has_wmh(self):
        self.assertFalse(_build_person().has_wmh())
        self.assertTrue(self._person_with_wmh(sbi=False, wmh=True).has_wmh())
        self.assertTrue(self._person_with_wmh(sbi=True, wmh=False).has_wmh())
        self.assertFalse(self._person_with_wmh(sbi=False, wmh=False).has_wmh())

    def test_get_scd_group(self):
        for sbi in [False, True]:
            for wmh in [False, True]:
                person = self._person_with_wmh(sbi=sbi, wmh=wmh)
                self.assertEqual(scdGroupMap[int(wmh)][int(sbi)], person.get_scd_group())

    def test_get_modality_group(self):
        for modality in [Modality.CT.value, Modality.MR.value, Modality.NO.value]:
            self.assertEqual(modalityGroupMap[modality], _build_person(modality=modality).get_modality_group())
        person = _build_person(modality="unknownModality")
        with self.assertRaises(RuntimeError):
            person.get_modality_group()

    def test_get_scd_by_modality_group(self):
        person = self._person_with_wmh(sbi=True, wmh=False, modality=Modality.MR.value)
        expected = modalityGroupMap[Modality.MR.value] * 4 + scdGroupMap[0][1]
        self.assertEqual(expected, person.get_scd_by_modality_group())

    def test_get_wmh_severity_group(self):
        self.assertEqual(wmhSeverityGroupMap['unknown'],
                         self._person_with_wmh(severityUnknown=True).get_wmh_severity_group())
        self.assertEqual(wmhSeverityGroupMap[WMHSeverity.MODERATE.value],
                         self._person_with_wmh(severity=WMHSeverity.MODERATE).get_wmh_severity_group())

    def test_get_wmh_severity_by_modality_group(self):
        person = self._person_with_wmh(severity=WMHSeverity.MILD, modality=Modality.CT.value)
        expected = modalityGroupMap[Modality.CT.value] * len(wmhSeverityGroupMap) + wmhSeverityGroupMap[WMHSeverity.MILD.value]
        self.assertEqual(expected, person.get_wmh_severity_by_modality_group())


# ==========================================================================
# 13. QALY and survival accounting
# ==========================================================================

class TestQalyAndSurvival(unittest.TestCase):

    def test_get_total_qalys(self):
        person = _build_person()
        person._outcomes[OutcomeType.QUALITYADJUSTED_LIFE_YEARS].append((60, QALYOutcome(False, False, 1.0)))
        person._outcomes[OutcomeType.QUALITYADJUSTED_LIFE_YEARS].append((61, QALYOutcome(False, False, 0.8)))
        self.assertAlmostEqual(1.8, person.get_total_qalys())

    def test_get_outcome_survival_info_without_event(self):
        person = _build_person()
        _set_waves(person, 3)
        self.assertEqual([4, 0], person.get_outcome_survival_info([OutcomeType.STROKE], personFunctionsList=None))

    def test_get_outcome_survival_info_with_event(self):
        person = _build_person()
        _set_waves(person, 3)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual([2, 1], person.get_outcome_survival_info([OutcomeType.STROKE], personFunctionsList=None))

    def test_get_outcome_survival_info_applies_person_functions(self):
        person = _build_person()
        _set_waves(person, 3)
        info = person.get_outcome_survival_info([OutcomeType.STROKE], personFunctionsList=[lambda x: x._current_age])
        self.assertEqual([4, 0, 63], info)

    def test_get_person_years_with_outcome_by_end_of_wave(self):
        person = _build_person()
        _set_waves(person, 3)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        person._outcomes[OutcomeType.STROKE].append(_stroke(63))
        self.assertEqual(0, person.get_person_years_with_outcome_by_end_of_wave(OutcomeType.STROKE, wave=0))
        self.assertEqual(1, person.get_person_years_with_outcome_by_end_of_wave(OutcomeType.STROKE, wave=2))
        self.assertEqual(2, person.get_person_years_with_outcome_by_end_of_wave(OutcomeType.STROKE, wave=3))

    def test_get_followup_person_years_by_end_of_wave(self):
        person = _build_person()
        _set_waves(person, 3)
        self.assertEqual(3, person.get_followup_person_years_by_end_of_wave([OutcomeType.STROKE], wave=2))
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual(2, person.get_followup_person_years_by_end_of_wave([OutcomeType.STROKE], wave=2))

    def test_get_first_incidence_at_risk_ages(self):
        person = _build_person()
        _set_waves(person, 3)
        self.assertEqual([60, 61, 62, 63], person.get_first_incidence_at_risk_ages(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual([60, 61], person.get_first_incidence_at_risk_ages(OutcomeType.STROKE))
        person._outcomes[OutcomeType.STROKE].insert(0, _prior_stroke())
        self.assertEqual([], person.get_first_incidence_at_risk_ages(OutcomeType.STROKE))

    def test_followup_pair_matches_separate_calls(self):
        for makeOutcomes in [lambda p: None,
                             lambda p: p._outcomes[OutcomeType.STROKE].append(_stroke(61)),
                             lambda p: p._outcomes[OutcomeType.STROKE].append(_prior_stroke()),
                             lambda p: p._outcomes[OutcomeType.STROKE].extend([_prior_stroke(), _stroke(62)])]:
            person = _build_person()
            _set_waves(person, 3)
            makeOutcomes(person)
            event, personYears = person.get_followup_event_and_person_years([OutcomeType.STROKE], wave=3)
            self.assertEqual(person.has_any_outcome_by_end_of_wave([OutcomeType.STROKE], wave=3), event)
            self.assertEqual(person.get_followup_person_years_by_end_of_wave([OutcomeType.STROKE], wave=3), personYears)

    def test_first_incidence_pair_matches_separate_calls(self):
        for makeOutcomes in [lambda p: None,
                             lambda p: p._outcomes[OutcomeType.STROKE].append(_stroke(61)),
                             lambda p: p._outcomes[OutcomeType.STROKE].append(_prior_stroke()),
                             lambda p: p._outcomes[OutcomeType.STROKE].extend([_prior_stroke(), _stroke(62)])]:
            person = _build_person()
            _set_waves(person, 3)
            makeOutcomes(person)
            eventAge, atRiskAges = person.get_first_incidence_event_age_and_at_risk_ages(OutcomeType.STROKE)
            self.assertEqual(person.get_first_incidence_age(OutcomeType.STROKE), eventAge)
            self.assertEqual(person.get_first_incidence_at_risk_ages(OutcomeType.STROKE), atRiskAges)

    def test_get_ages_with_and_without_outcome(self):
        person = _build_person()
        _set_waves(person, 2)
        person._outcomes[OutcomeType.STROKE].append(_stroke(61))
        self.assertEqual([61], person.get_ages_with_outcome(OutcomeType.STROKE))
        self.assertEqual({60, 62}, set(person.get_ages_without_outcome(OutcomeType.STROKE)))


# ==========================================================================
# 14. Glucose conversions
# ==========================================================================

class TestFastingGlucose(unittest.TestCase):

    def test_without_residual_is_deterministic_conversion(self):
        person = _build_person(a1c=5.5)
        self.assertEqual(convert_a1c_to_fasting_glucose(5.5), person.get_fasting_glucose(use_residual=False))

    def test_with_residual_draws_from_person_rng(self):
        person = _build_person(a1c=5.5)
        person._rng = np.random.default_rng(7215)
        expected = convert_a1c_to_fasting_glucose(5.5) + np.random.default_rng(7215).normal(0, 21)
        self.assertAlmostEqual(expected, person.get_fasting_glucose(use_residual=True))


# ==========================================================================
# 15. Eligibility filters
# ==========================================================================

class TestAllhatCandidate(unittest.TestCase):

    def test_candidate(self):
        person = _build_person(age=60, sbp=150, dbp=95, hdl=30)
        self.assertTrue(person.allhat_candidate(0))

    def test_not_candidate_without_qualifying_condition(self):
        person = _build_person(age=60, sbp=150, dbp=95, hdl=50)  # bp fits but no smoking/a1c/stroke/mi/low hdl
        self.assertFalse(person.allhat_candidate(0))

    def test_not_candidate_by_bp(self):
        person = _build_person(age=60, sbp=120, dbp=95, hdl=30)
        self.assertFalse(person.allhat_candidate(0))


# ==========================================================================
# 16. State export
# ==========================================================================

class TestStateExport(unittest.TestCase):

    def test_get_current_state_as_dict(self):
        person = _build_person()
        _set_waves(person, 2)
        state = person.get_current_state_as_dict()
        self.assertEqual("testPerson", state["name"])
        self.assertEqual(NHANESGender.MALE, state["gender"])
        self.assertEqual(62, state["age"])  # last value only
        self.assertEqual(0, state["statin"])
        self.assertIs(person._outcomes, state["outcomes"])

    def test_get_full_state_as_dict(self):
        person = _build_person()
        _set_waves(person, 2)
        state = person.get_full_state_as_dict()
        self.assertEqual([60, 61, 62], state["age"])  # full history
        self.assertEqual([0, 0, 0], state["statin"])


# ==========================================================================
# 17. Dunder methods
# ==========================================================================

class TestDunderMethods(unittest.TestCase):

    def test_repr_smoke(self):
        person = _build_person()
        _set_waves(person, 2)
        personRepr = repr(person)
        self.assertIsInstance(personRepr, str)
        self.assertIn("testPerson", personRepr)

    def test_repr_handles_none_dynamic_risk_factor(self):
        person = _build_person()
        person._afib = [None]  # half-initialized person, eg before the afib model has run
        self.assertIn("afib=None", repr(person))


    def test_eq_for_identical_persons(self):
        self.assertEqual(_build_person(), _build_person())

    def test_eq_detects_differences(self):
        self.assertNotEqual(_build_person(sbp=120), _build_person(sbp=130))
        personA, personB = _build_person(), _build_person()
        personB._treatmentStrategies[BP]["status"] = TreatmentStrategyStatus.BEGIN
        self.assertNotEqual(personA, personB)
        personC = _build_person()
        personC._waveCompleted = 0
        self.assertNotEqual(personA, personC)

    def test_ne(self):
        self.assertTrue(_build_person(sbp=120) != _build_person(sbp=130))
        self.assertFalse(_build_person() != _build_person())

    def test_hash_is_stable_and_state_independent(self):
        person = _build_person()
        hashBefore = hash(person)
        _set_waves(person, 3)
        self.assertEqual(hashBefore, hash(person))
        self.assertEqual(hash(_build_person()), hash(_build_person()))  # same name and index

    def test_deepcopy_at_baseline(self):
        person = _build_person()
        personCopy = person.__deepcopy__()
        self.assertIsNot(person, personCopy)
        self.assertEqual(person, personCopy)
        personCopy._sbp[0] = 130
        self.assertEqual(120, person._sbp[0])  # copy is independent

    def test_deepcopy_after_advance_raises(self):
        person = _build_person()
        _set_waves(person, 1)
        with self.assertRaises(RuntimeError):
            person.__deepcopy__()


if __name__ == "__main__":
    unittest.main()
