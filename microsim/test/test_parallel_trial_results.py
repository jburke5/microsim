"""Trial results must not depend on whether the trial's people were created serially or in
parallel. Nothing is seeded, so two trials never agree exactly: they are compared within their
combined standard errors, z = (a - b) / sqrt(seA^2 + seB^2), and a sensitivity test shows the
comparison does flag a real difference.

Only creation differs: the serial trial is built with nWorkers=1 and then advanced with the same
workers as the parallel one.

Slow: builds and runs several trials."""

import math
import unittest

import numpy as np

from microsim.outcomes.outcome import OutcomeType
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.risk_factors.gender import NHANESGender
from microsim.trials.trial import Trial
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial_outcome_assessor import AnalysisType
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
from microsim.trials.trial_type import TrialType

ADVANCE_WORKERS = 4
# 45 comparisons (14 baseline, 18 risk, 13 effect); 20 serial-vs-serial runs peaked at
# max|z| = 3.24, mostly 1.5-2.5, so 4 leaves headroom without being toothless
Z_MAX = 4.0

BASELINE = {
    "age": lambda p: p._age[0],
    "sbp": lambda p: p._sbp[0],
    "ldl": lambda p: p._ldl[0],
    "bmi": lambda p: p._bmi[0],
    "female": lambda p: p._gender == NHANESGender.FEMALE,
    "priorStroke": lambda p: p.has_outcome_prior_to_simulation(OutcomeType.STROKE),
    "priorMI": lambda p: p.has_outcome_prior_to_simulation(OutcomeType.MI),
}


def build_trial(
    nWorkers, minAge=60, sampleSize=2000, duration=5, trialType=TrialType.COMPLETELY_RANDOMIZED
):
    """Creates the people with nWorkers, then runs and analyzes with ADVANCE_WORKERS."""
    pf = PersonFilterFactory.get_person_filter(["adult"])
    pf.add_filter("person", "minAge", lambda x: x._age[0] >= minAge)
    trial = Trial(
        NhanesTrialDescription(
            trialType=trialType,
            sampleSize=sampleSize,
            duration=duration,
            treatmentStrategies="1bpMedsAdded",
            nWorkers=nWorkers,
            personFilters=pf,
            year=1999,
            nhanesWeights=True,
        ),
        notify=False,
    )
    trial.trialDescription.nWorkers = ADVANCE_WORKERS
    trial.run_analyze(TrialOutcomeAssessorFactory.get_trial_outcome_assessor(), notify=False)
    return trial


def _z(a, seA, b, seB):
    se = math.sqrt(seA**2 + seB**2)
    return None if se == 0 else (a - b) / se


def baseline_z_scores(trialA, trialB):
    z = {}
    for arm in ("treatedPop", "controlPop"):
        peopleA = getattr(trialA, arm)._people
        peopleB = getattr(trialB, arm)._people
        for name, f in BASELINE.items():
            a = np.array([f(p) for p in peopleA], dtype=float)
            b = np.array([f(p) for p in peopleB], dtype=float)
            z[f"{arm}.{name}"] = _z(
                a.mean(),
                a.std(ddof=1) / math.sqrt(len(a)),
                b.mean(),
                b.std(ddof=1) / math.sqrt(len(b)),
            )
    return z


def risk_z_scores(trialA, trialB):
    """tRisk/cRisk are event counts over arm size; recurrent events make the variance Poisson
    rather than binomial, sqrt(count)/n, which is also the more conservative of the two."""
    z = {}
    for name, resA in trialA.results[AnalysisType.RELATIVE_RISK.value].items():
        resB = trialB.results[AnalysisType.RELATIVE_RISK.value][name]
        for label, i, pop in (("tRisk", 3, "treatedPop"), ("cRisk", 8, "controlPop")):
            nA, nB = getattr(trialA, pop)._n, getattr(trialB, pop)._n
            rA, rB = resA[i], resB[i]
            z[f"{name}.{label}"] = _z(rA, math.sqrt(rA / nA), rB, math.sqrt(rB / nB))
    return z


def effect_z_scores(trialA, trialB):
    z = {}
    for analysisType in (AnalysisType.LOGISTIC, AnalysisType.LINEAR, AnalysisType.COX):
        for name, resA in trialA.results[analysisType.value].items():
            resB = trialB.results[analysisType.value][name]
            values = (resA[0], resA[1], resB[0], resB[1])
            # a failed fit reports NaN, see the regression analyses
            if any(v is None or not np.isfinite(v) for v in values):
                z[f"{analysisType.value}.{name}"] = None
            else:
                z[f"{analysisType.value}.{name}"] = _z(*values)
    return z


class _ZScoreAssertions(unittest.TestCase):
    def assert_agree(self, z):
        skipped = sorted(k for k, v in z.items() if v is None or not np.isfinite(v))
        scored = {k: v for k, v in z.items() if k not in skipped}
        self.assertGreater(len(scored), 0, f"nothing to compare, skipped: {skipped}")
        largest = sorted(scored.items(), key=lambda kv: -abs(kv[1]))[:5]
        self.assertLess(
            abs(largest[0][1]),
            Z_MAX,
            f"largest |z|: {[(k, round(v, 2)) for k, v in largest]}, skipped: {skipped}",
        )


class TestSerialAndParallelTrialsAgree(_ZScoreAssertions):
    @classmethod
    def setUpClass(cls):
        cls.serial = build_trial(nWorkers=1)
        cls.parallel = build_trial(nWorkers=ADVANCE_WORKERS)

    def test_arms_have_the_same_sizes(self):
        for trial in (self.serial, self.parallel):
            self.assertEqual(2000, len(trial.treatedPop._people))
            self.assertEqual(2000, len(trial.controlPop._people))

    def test_eligibility_holds_in_both(self):
        for trial in (self.serial, self.parallel):
            for pop in (trial.treatedPop, trial.controlPop):
                self.assertGreaterEqual(min(p._age[0] for p in pop._people), 60)

    def test_indexes_are_unique_in_both(self):
        for trial in (self.serial, self.parallel):
            indexes = [
                p._index for pop in (trial.treatedPop, trial.controlPop) for p in pop._people
            ]
            self.assertEqual(len(indexes), len(set(indexes)))

    def test_baseline_covariates_agree(self):
        self.assert_agree(baseline_z_scores(self.serial, self.parallel))

    def test_event_risks_agree(self):
        self.assert_agree(risk_z_scores(self.serial, self.parallel))

    def test_treatment_effects_agree(self):
        self.assert_agree(effect_z_scores(self.serial, self.parallel))

    def test_the_comparison_detects_a_real_difference(self):
        older = build_trial(nWorkers=1, minAge=65, sampleSize=1000, duration=1)
        z = baseline_z_scores(self.serial, older)
        self.assertGreater(max(abs(v) for v in z.values() if v is not None), Z_MAX)


class TestPotentialOutcomesTrialAgrees(_ZScoreAssertions):
    @classmethod
    def setUpClass(cls):
        args = dict(sampleSize=1000, duration=2, trialType=TrialType.POTENTIAL_OUTCOMES)
        cls.serial = build_trial(nWorkers=1, **args)
        cls.parallel = build_trial(nWorkers=ADVANCE_WORKERS, **args)

    def test_baseline_covariates_agree(self):
        self.assert_agree(baseline_z_scores(self.serial, self.parallel))

    def test_arms_start_identical(self):
        treated, control = self.parallel.treatedPop._people, self.parallel.controlPop._people
        self.assertEqual([p._name for p in treated], [p._name for p in control])
        # 1bpMedsAdded lowers the treated arm's first-wave sbp, so sbp is left out
        for name, f in BASELINE.items():
            if name != "sbp":
                self.assertEqual([f(p) for p in treated], [f(p) for p in control], name)


if __name__ == "__main__":
    unittest.main()
