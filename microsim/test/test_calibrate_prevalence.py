import unittest

from microsim.common.age_scope import AgeScope
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository
from microsim.population.population_factory import PopulationFactory
from microsim.common.population_type import PopulationType
from microsim.person.person_filter_factory import PersonFilterFactory


def _adults_filter():
    return PersonFilterFactory.get_person_filter(["adult"])


def _nhanes_args(n=500):
    """Weighted-sampling args, for refusal tests where we don't care about reproducibility."""
    return {"n": n, "year": 1999, "personFilters": _adults_filter(),
            "nhanesWeights": True, "distributions": False}


def _deterministic_nhanes_args():
    """Use the entire filtered NHANES dataset (no sampling). Deterministic across calls,
       so the calibrator and any test-side recompute see the exact same persons."""
    return {"n": None, "year": 1999, "personFilters": _adults_filter(),
            "nhanesWeights": False, "distributions": False}


class TestAgeScope(unittest.TestCase):
    def test_exact_age_matches(self):
        scope = AgeScope(75, 75)
        self.assertTrue(scope.contains(75))
        self.assertFalse(scope.contains(74))
        self.assertFalse(scope.contains(76))

    def test_age_group_inclusive_both_ends(self):
        scope = AgeScope(70, 74)
        for age in (70, 71, 72, 73, 74):
            self.assertTrue(scope.contains(age))
        self.assertFalse(scope.contains(69))
        self.assertFalse(scope.contains(75))

    def test_pooled_65_plus(self):
        scope = AgeScope(lo=65)
        self.assertFalse(scope.contains(64))
        self.assertTrue(scope.contains(65))
        self.assertTrue(scope.contains(120))

    def test_pooled_overall_accepts_everything(self):
        scope = AgeScope()
        for age in (0, 18, 65, 95):
            self.assertTrue(scope.contains(age))

    def test_lo_greater_than_hi_raises(self):
        with self.assertRaises(ValueError):
            AgeScope(75, 70)

    def test_labels(self):
        self.assertEqual(AgeScope().label, "pooled_overall")
        self.assertEqual(AgeScope(lo=65).label, "pooled_65_plus")
        self.assertEqual(AgeScope(70, 74).label, "age_group_70-74")
        self.assertEqual(AgeScope(75, 75).label, "age_75")


def _measure_realized_prev(scaleOutcomeType, targetOutcomeType, scaling, scope, peopleArgs,
                           baselineRiskScaling=None, rebuilds=5):
    """Build people `rebuilds` times with the chosen scaling and return the mean realized
       priorToSim prevalence of targetOutcomeType in scope.
       Deterministic peopleArgs select the same persons every build, but each build gives every
       Person a fresh unseeded RNG (Person.__init__), and the prevalence models draw from it, so
       a single build carries about one binomial standard error of seeding noise. Averaging over
       rebuilds shrinks that by sqrt(rebuilds)."""
    rs = dict(baselineRiskScaling or {})
    rs[scaleOutcomeType] = scaling
    #useDefaults=False to mirror calibrate_prevalence, which measures against a pristine baseline
    opmr = OutcomePrevalenceModelRepository(riskScaling=rs, useDefaults=False)
    prevalences = []
    for _ in range(rebuilds):
        people = PopulationFactory.get_nhanes_people(
            **peopleArgs, outcomePrevalenceModelRepository=opmr
        )
        inScopePeople = [p for p in people if scope.contains(p._current_age)]
        hits = sum(1 for p in inScopePeople if p.has_outcome_prior_to_simulation(targetOutcomeType))
        prevalences.append(hits / len(inScopePeople))
    return sum(prevalences) / len(prevalences)


class TestCalibratePrevalenceRefusals(unittest.TestCase):
    def test_kaiser_pop_type_refused(self):
        with self.assertRaises(NotImplementedError):
            PopulationFactory.calibrate_prevalence(
                scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
                targetOutcomeType=OutcomeType.STROKE,
                target=0.05, scope=AgeScope(),
                popType=PopulationType.KAISER, peopleArgs=_nhanes_args(),
            )

    def test_scale_mi_refused(self):
        with self.assertRaises(ValueError):
            PopulationFactory.calibrate_prevalence(
                scaleOutcomeType=OutcomeType.MI,
                targetOutcomeType=OutcomeType.MI,
                target=0.05, scope=AgeScope(),
                popType=PopulationType.NHANES, peopleArgs=_nhanes_args(),
            )

    def test_scale_cognition_refused(self):
        with self.assertRaises(ValueError):
            PopulationFactory.calibrate_prevalence(
                scaleOutcomeType=OutcomeType.COGNITION,
                targetOutcomeType=OutcomeType.COGNITION,
                target=0.05, scope=AgeScope(),
                popType=PopulationType.NHANES, peopleArgs=_nhanes_args(),
            )

    def test_scale_without_prevalence_model_refused(self):
        with self.assertRaises(ValueError):
            PopulationFactory.calibrate_prevalence(
                scaleOutcomeType=OutcomeType.DEATH,
                targetOutcomeType=OutcomeType.STROKE,
                target=0.05, scope=AgeScope(),
                popType=PopulationType.NHANES, peopleArgs=_nhanes_args(),
            )

    def test_target_without_prevalence_model_refused(self):
        with self.assertRaises(ValueError):
            PopulationFactory.calibrate_prevalence(
                scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
                targetOutcomeType=OutcomeType.DEATH,
                target=0.05, scope=AgeScope(),
                popType=PopulationType.NHANES, peopleArgs=_nhanes_args(),
            )

    def test_target_before_scale_refused(self):
        # STROKE precedes DEMENTIA in OutcomeType order — scaling DEMENTIA cannot
        # cascade backwards to STROKE.
        with self.assertRaises(ValueError):
            PopulationFactory.calibrate_prevalence(
                scaleOutcomeType=OutcomeType.DEMENTIA,
                targetOutcomeType=OutcomeType.STROKE,
                target=0.05, scope=AgeScope(),
                popType=PopulationType.NHANES, peopleArgs=_nhanes_args(),
            )

    def test_target_out_of_range_refused(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                PopulationFactory.calibrate_prevalence(
                    scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
                    targetOutcomeType=OutcomeType.STROKE,
                    target=bad, scope=AgeScope(),
                    popType=PopulationType.NHANES, peopleArgs=_nhanes_args(),
                )


class TestCalibratePrevalenceHitsTarget(unittest.TestCase):
    """Cross-outcome calibration: drive realized priorToSim target prevalence to target
       by scaling a precursor outcome. Stroke and MI both gate on prior CV in their
       prevalence models, so scaling CV cascades into both."""

    def test_cv_scale_stroke_target(self):
        target = 0.05
        scope = AgeScope()
        peopleArgs = _deterministic_nhanes_args()
        scaling = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
            targetOutcomeType=OutcomeType.STROKE,
            target=target, scope=scope,
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        realized = _measure_realized_prev(
            OutcomeType.CARDIOVASCULAR, OutcomeType.STROKE, scaling, scope, peopleArgs
        )
        self.assertAlmostEqual(realized, target, delta=0.01)

    def test_cv_scale_mi_target(self):
        target = 0.04
        scope = AgeScope()
        peopleArgs = _deterministic_nhanes_args()
        scaling = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
            targetOutcomeType=OutcomeType.MI,
            target=target, scope=scope,
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        realized = _measure_realized_prev(
            OutcomeType.CARDIOVASCULAR, OutcomeType.MI, scaling, scope, peopleArgs
        )
        self.assertAlmostEqual(realized, target, delta=0.01)


class TestCalibratePrevalenceSameOutcome(unittest.TestCase):
    """Smoke: scale == target. Should drive realized prevalence to target."""

    def test_dementia_same_outcome_hits_target(self):
        target = 0.10
        scope = AgeScope(lo=65)
        peopleArgs = _deterministic_nhanes_args()
        scaling = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.DEMENTIA,
            targetOutcomeType=OutcomeType.DEMENTIA,
            target=target, scope=scope,
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        realized = _measure_realized_prev(
            OutcomeType.DEMENTIA, OutcomeType.DEMENTIA, scaling, scope, peopleArgs
        )
        #the 65+ scope holds ~1392 persons, so seeding noise alone is ~0.008 here: the calibrator
        #solves against one frozen set of draws (see calibrate_prevalence) and this measurement
        #uses fresh ones. 0.01 would fail about a quarter of the time even when calibration is correct.
        self.assertAlmostEqual(realized, target, delta=0.02)


class TestCalibratePrevalenceScopeMatters(unittest.TestCase):
    """Use CV — its linear predictor depends on age, so 65+ vs overall have
       materially different baseline distributions and require different scalings
       to hit the same target. (Dementia would fail here because its placeholder
       coefficients are all zero, so lp is constant across age subsets.)"""

    def test_scope_affects_scaling_for_age_dependent_model(self):
        peopleArgs = _deterministic_nhanes_args()
        scaling65 = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
            targetOutcomeType=OutcomeType.CARDIOVASCULAR,
            target=0.10, scope=AgeScope(lo=65),
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        scalingAll = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
            targetOutcomeType=OutcomeType.CARDIOVASCULAR,
            target=0.10, scope=AgeScope(),
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        self.assertNotAlmostEqual(scaling65, scalingAll, places=2)


class TestCalibratePrevalenceMonotone(unittest.TestCase):
    """Higher target should require a larger scaling (for an outcome where the
       prevalence model is monotone in its riskScaling — true of all expit-based
       models, which dementia is)."""

    def test_dementia_higher_target_requires_larger_scaling(self):
        peopleArgs = _deterministic_nhanes_args()
        scope = AgeScope(lo=65)
        scalingLo = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.DEMENTIA,
            targetOutcomeType=OutcomeType.DEMENTIA,
            target=0.05, scope=scope,
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        scalingHi = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.DEMENTIA,
            targetOutcomeType=OutcomeType.DEMENTIA,
            target=0.20, scope=scope,
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        self.assertLess(scalingLo, scalingHi)


class TestCalibratePrevalenceDropsOpmrFromPeopleArgs(unittest.TestCase):
    """If the caller accidentally passes outcomePrevalenceModelRepository inside
       peopleArgs (e.g. by reusing a NhanesTrialDescription.peopleArgs dict), the
       calibrator must drop it rather than crash with a duplicate-kwarg TypeError."""

    def test_opmr_in_peopleargs_is_silently_dropped(self):
        peopleArgs = _deterministic_nhanes_args()
        peopleArgs["outcomePrevalenceModelRepository"] = OutcomePrevalenceModelRepository()
        scaling = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.DEMENTIA,
            targetOutcomeType=OutcomeType.DEMENTIA,
            target=0.10, scope=AgeScope(lo=65),
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        self.assertGreater(scaling, 0)


class TestCalibratePrevalenceNoCascade(unittest.TestCase):
    """If scaling has no downstream effect on the target, the brentq bracket gap
       won't change sign and the calibrator must raise rather than return a bogus value."""

    def test_no_cascade_refused(self):
        # DEMENTIA precedes EPILEPSY in OutcomeType order, but EpilepsyPrevalenceModel
        # doesn't reference dementia status, so scaling dementia leaves epilepsy
        # prevalence untouched.
        with self.assertRaises(ValueError):
            PopulationFactory.calibrate_prevalence(
                scaleOutcomeType=OutcomeType.DEMENTIA,
                targetOutcomeType=OutcomeType.EPILEPSY,
                target=0.10, scope=AgeScope(),
                popType=PopulationType.NHANES,
                peopleArgs=_deterministic_nhanes_args(),
            )


class TestCalibratePrevalenceNhanesWeights(unittest.TestCase):
    """nhanesWeights=True is the user-facing reason this function exists. The inner
       loop is deterministic in s (rng snapshot/restore) so brentq converges even
       though the sample itself was drawn stochastically."""

    def test_cv_scale_stroke_target_weighted_sample(self):
        # Target stays well below the racial-mix ceiling baked into the placeholder
        # StrokePrevalenceModel coefficients (NHW has -27.3 in lp, capping reachable
        # stroke prevalence even with CV scaling maxed out).
        target = 0.02
        scope = AgeScope()
        peopleArgs = _nhanes_args(n=1000)
        scaling = PopulationFactory.calibrate_prevalence(
            scaleOutcomeType=OutcomeType.CARDIOVASCULAR,
            targetOutcomeType=OutcomeType.STROKE,
            target=target, scope=scope,
            popType=PopulationType.NHANES, peopleArgs=peopleArgs,
        )
        self.assertGreater(scaling, 0)


if __name__ == "__main__":
    unittest.main()
