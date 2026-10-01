"""Tests pinning the fixes from the risk_factors package review:
Asian-race handling in spec-driven models, person-aware risk factor bounds,
the absorbing PVD state, GFR equation error paths, and the alcohol prevalence model."""

import contextlib
import io
import math
import unittest

import numpy as np
import pandas as pd

from microsim.common.data_loader import load_regression_model
from microsim.person.person import Person
from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel
from microsim.regression_models.regression_model import RegressionModel
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.alcohol_model import AlcoholPrevalenceModel
from microsim.risk_factors.cohort_risk_model_repository import CohortDynamicRiskFactorModelRepository
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.gfr_equation import GFREquation
from microsim.risk_factors.nhanes_linear_risk_factor_model import NHANESLinearRiskFactorModel
from microsim.risk_factors.pvd_model import PVDIncidenceModel
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.risk_factor_bounds import RiskFactorBounds
from microsim.risk_factors.risk_model_repository import BoundedRiskFactorModel
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.test.test_person_functions import _build_person


class _Strategy:
    def __init__(self, updatedRiskFactors):
        self._updatedRiskFactors = updatedRiskFactors

    def get_updated_risk_factors(self, person):
        return dict(self._updatedRiskFactors)


# ==========================================================================
# Asian race maps to white in every spec-driven model
# ==========================================================================

class TestAsianRaceHandling(unittest.TestCase):

    def test_asian_equals_white_for_indicator_coefficients(self):
        model = LinearRiskFactorModel(load_regression_model("hdlCohortModel"))
        white = _build_person(raceEthnicity=RaceEthnicity.NON_HISPANIC_WHITE)
        asian = _build_person(raceEthnicity=RaceEthnicity.ASIAN)
        black = _build_person(raceEthnicity=RaceEthnicity.NON_HISPANIC_BLACK)
        self.assertEqual(model.estimate_next_risk(white), model.estimate_next_risk(asian))
        self.assertNotEqual(model.estimate_next_risk(white), model.estimate_next_risk(black))

    def test_asian_equals_white_for_raw_race_coefficient(self):
        model = LinearRiskFactorModel(RegressionModel({"Intercept": 0, "raceEthnicity": 1.0}, {}, 0, 0))
        asian = _build_person(raceEthnicity=RaceEthnicity.ASIAN)
        self.assertEqual(RaceEthnicity.NON_HISPANIC_WHITE.value, model.estimate_next_risk(asian))

    def test_person_race_is_not_modified(self):
        model = LinearRiskFactorModel(load_regression_model("hdlCohortModel"))
        asian = _build_person(raceEthnicity=RaceEthnicity.ASIAN)
        model.estimate_next_risk(asian)
        self.assertEqual(RaceEthnicity.ASIAN, asian._raceEthnicity)

    def test_asian_equals_white_in_legacy_nhanes_model(self):
        params = {"age": 0, "gender": 0, "raceEthnicity[T.2]": 1, "raceEthnicity[T.3]": 5,
                  "raceEthnicity[T.4]": 2, "raceEthnicity[T.5]": 3, "smokingStatus[T.1]": 0,
                  "smokingStatus[T.2]": 0, "sbp": 0, "dbp": 0, "a1c": 0, "hdl": 0,
                  "totChol": 0, "bmi": 0, "Intercept": 10}
        model = NHANESLinearRiskFactorModel(params=params, resids=pd.Series(np.zeros(10)))
        estimate = lambda race: model.estimate_risk_for_params(
            60, 1, 120, 80, 5.5, 50, 200, 25, race, SmokingStatus.NEVER, rng=np.random.default_rng(0))
        self.assertEqual(15, estimate(RaceEthnicity.NON_HISPANIC_WHITE))
        self.assertEqual(15, estimate(RaceEthnicity.ASIAN))
        self.assertEqual(10, estimate(RaceEthnicity.MEXICAN_AMERICAN))

    def test_asian_and_white_persons_advance_identically(self):
        repo = CohortDynamicRiskFactorModelRepository()
        white = _build_person(raceEthnicity=RaceEthnicity.NON_HISPANIC_WHITE)
        asian = _build_person(raceEthnicity=RaceEthnicity.ASIAN)
        white._rng = np.random.default_rng(7)
        asian._rng = np.random.default_rng(7)
        white.advance_risk_factors(repo)
        asian.advance_risk_factors(repo)
        for rf in white._dynamicRiskFactors:
            self.assertEqual(getattr(white, "_"+rf)[-1], getattr(asian, "_"+rf)[-1], rf)


# ==========================================================================
# Person-aware risk factor bounds
# ==========================================================================

class TestPersonAwareBounds(unittest.TestCase):

    def test_apply_to_person_uses_person_age(self):
        adult = _build_person(age=60)
        child = _build_person(age=10)
        self.assertEqual(297., RiskFactorBounds.apply_to_person("sbp", 500, adult))
        self.assertEqual(190.3, RiskFactorBounds.apply_to_person("sbp", 500, child))

    def test_age_is_judged_by_its_proposed_next_value(self):
        seventeen = _build_person(age=17)
        self.assertEqual(18, RiskFactorBounds.apply_to_person("age", 18, seventeen))

    def test_repository_wraps_models_with_bounds(self):
        repo = CohortDynamicRiskFactorModelRepository()
        self.assertIsInstance(repo.get_model("sbp"), BoundedRiskFactorModel)

    def test_update_risk_factors_applies_child_bounds(self):
        child = _build_person(age=10)
        child.update_risk_factors(_Strategy({"sbp": 500}))
        self.assertEqual(190.3, child._sbp[-1])


# ==========================================================================
# PVD is an absorbing state
# ==========================================================================

class TestPVDIsAbsorbing(unittest.TestCase):

    def test_lag_pvd_gives_risk_of_exactly_one(self):
        model = PVDIncidenceModel()
        lp = model.calc_linear_predictor_for_patient_characteristics(
            70, 140, 80, 200, 50, NHANESGender.MALE, SmokingStatus.NEVER,
            RaceEthnicity.NON_HISPANIC_WHITE, True)
        self.assertGreater(lp, 10)  # above the clamp, so the risk is exactly 1

    def test_person_with_pvd_always_keeps_it(self):
        model = PVDIncidenceModel()
        person = _build_person()
        person._pvd = [True]
        for _ in range(100):
            self.assertTrue(model.estimate_next_risk(person))


# ==========================================================================
# GFR equation
# ==========================================================================

class TestGFREquation(unittest.TestCase):

    def test_wave_argument_selects_the_wave(self):
        person = _build_person(age=60)
        person._age = [60, 61]
        person._creatinine = [0.9, 1.5]
        gfr = GFREquation()
        self.assertEqual(gfr.get_gfr_for_person_attributes(
            person._gender, person._raceEthnicity, 0.9, 60), gfr.get_gfr_for_person(person, wave=0))
        self.assertEqual(gfr.get_gfr_for_person_attributes(
            person._gender, person._raceEthnicity, 1.5, 61), gfr.get_gfr_for_person(person))

    def test_diagnostic_path_does_not_raise(self):
        # nan creatinine fires the diagnostic print, which used to raise a NameError
        with contextlib.redirect_stdout(io.StringIO()):
            result = GFREquation().get_gfr_for_person_attributes(
                NHANESGender.FEMALE, RaceEthnicity.NON_HISPANIC_BLACK, np.nan, 60)
        self.assertTrue(math.isnan(result))


# ==========================================================================
# Alcohol prevalence model
# ==========================================================================

class TestAlcoholPrevalenceModel(unittest.TestCase):

    def test_onetosix_has_zero_probability_by_design(self):
        # NHANES has no ONETOSIX rows, so the first two polr intercepts are deliberately equal
        lps = AlcoholPrevalenceModel().calc_linear_predictor_for_patient_characteristics(
            NHANESGender.MALE, SmokingStatus.NEVER, 60)
        self.assertEqual(lps[0], lps[1])

    def test_estimate_never_returns_onetosix(self):
        model = AlcoholPrevalenceModel()
        person = _build_person(age=45)
        draws = [model.estimate_next_risk(person) for _ in range(300)]
        self.assertNotIn(AlcoholCategory.ONETOSIX, draws)
        self.assertGreater(len(set(draws)), 1)  # the other categories do occur

    def test_error_messages_name_the_right_model(self):
        model = AlcoholPrevalenceModel()
        with self.assertRaisesRegex(RuntimeError, "gender in AlcoholPrevalenceModel"):
            model.calc_linear_predictor_for_patient_characteristics("bad", SmokingStatus.NEVER, 60)
        with self.assertRaisesRegex(RuntimeError, "smokingStatus in AlcoholPrevalenceModel"):
            model.calc_linear_predictor_for_patient_characteristics(NHANESGender.MALE, "bad", 60)


if __name__ == "__main__":
    unittest.main()
