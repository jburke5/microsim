"""Tests for building NHANES people in parallel (get_nhanes_people(..., nWorkers>1)).

They assume this API:
- get_nhanes_people / get_nhanes_population(..., nWorkers=1): nWorkers=1 is the serial path; the
  pool is a plain multiprocessing.Pool, so the tests pick fork or spawn with _start_method.
- PopulationFactory.split_draws(n, maxDraws, nWorkers): one (n, maxDraws) share per worker.
- PopulationFactory.get_nhanes_draw_tasks(nWorkers, **getNhanesPeopleArgs): one cloudpickled task
  per worker, holding everything the worker needs (rows, weights, filters, distributions, caches,
  its share, its seed).
- PopulationFactory.draw_people_worker(task): returns (people, drawn, accepted) for one task.
- Chunks are concatenated in worker order, so the first split_draws share is worker 0's people.

Slow: most of these build pools of worker processes."""

import contextlib
import copy
import multiprocessing
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from microsim.common.population_type import PopulationType
from microsim.outcomes.outcome import OutcomeType
from microsim.person.person import Person
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.population.population_factory import PopulationFactory
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType
from microsim.trials.trial import Trial
from microsim.trials.trial_description import KaiserTrialDescription, NhanesTrialDescription
from microsim.trials.trial_factory import TrialFactory
from microsim.trials.trial_type import TrialType

REPO_ROOT = Path(__file__).resolve().parents[2]
HAS_FORK = "fork" in multiprocessing.get_all_start_methods()

CONTINUOUS = ["_sbp", "_dbp", "_a1c", "_hdl", "_ldl", "_trig", "_totChol", "_bmi", "_waist"]


def _adults_with_person_filter(name, filterFunction):
    pf = PersonFilterFactory.get_person_filter(["adult"])
    pf.add_filter("person", name, filterFunction)
    return pf


def _chunks(people, n, nWorkers):
    """The people of each worker, by the shares split_draws gives them."""
    shares = [s[0] for s in PopulationFactory.split_draws(n, max(100 * n, 500), nWorkers)]
    bounds = np.cumsum([0] + shares)
    return [people.iloc[bounds[i] : bounds[i + 1]] for i in range(len(shares))]


@contextlib.contextmanager
def _start_method(method):
    """Sets the process-wide start method, which a plain multiprocessing.Pool uses, and restores
    it."""
    saved = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method(method, force=True)
    try:
        yield
    finally:
        multiprocessing.set_start_method(saved, force=True)


def _run_script(body, timeout=600):
    """Runs body as a guarded script in a fresh interpreter: worker output and hangs are only
    observable from outside the process."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "script.py"
        path.write_text('if __name__ == "__main__":\n' + "".join(f"    {line}\n" for line in body))
        return subprocess.run(
            [sys.executable, str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


class _RestoreCaches(unittest.TestCase):
    CACHES = [
        "_nhanesDf",
        "_nhanesDfResampled",
        "_crudeDistributions",
        "_groupMeans",
        "_yearCorrections",
    ]

    def setUp(self):
        self._saved = {c: getattr(PopulationFactory, c) for c in self.CACHES}

    def tearDown(self):
        for c, v in self._saved.items():
            setattr(PopulationFactory, c, v)


class TestWorkersDrawIndependently(unittest.TestCase):
    """Forked workers inherit the parent's global NumPy state, so without their own seeds every
    worker would draw the same rows."""

    def _assert_chunks_differ(self, startMethod):
        n = 200
        with _start_method(startMethod):
            people = PopulationFactory.get_nhanes_people(
                n=n, year=1999, nhanesWeights=True, nWorkers=2
            )
        first, second = _chunks(people, n, 2)
        self.assertNotEqual([p._name for p in first], [p._name for p in second][: len(first)])

    @unittest.skipUnless(HAS_FORK, "fork is not available on this platform")
    def test_forked_workers_draw_different_rows(self):
        self._assert_chunks_differ("fork")

    def test_spawned_workers_draw_different_rows(self):
        self._assert_chunks_differ("spawn")

    def test_no_two_people_share_their_redrawn_variables(self):
        people = PopulationFactory.get_nhanes_people(
            n=100, year=1999, nhanesWeights=True, distributions=True, nWorkers=2
        )
        profiles = {tuple(getattr(p, v)[0] for v in CONTINUOUS) for p in people}
        self.assertEqual(len(people), len(profiles))

    def test_person_rngs_are_not_duplicated(self):
        people = PopulationFactory.get_nhanes_people(n=200, year=1999, nWorkers=2)
        self.assertEqual(len(people), len({p._rng.uniform() for p in people}))

    def test_tasks_draw_differently_from_the_same_inherited_state(self):
        tasks = PopulationFactory.get_nhanes_draw_tasks(
            2, n=40, year=1999, personFilters=None, nhanesWeights=True
        )
        names = []
        for task in tasks:
            np.random.seed(0)  # what every forked worker would start from
            people, _, _ = PopulationFactory.draw_people_worker(task)
            names.append([p._name for p in people])
        self.assertNotEqual(names[0], names[1])


class TestWorkersNeedNoCaches(_RestoreCaches):
    """Spawned workers start with empty caches, and rebuilding them costs ~19 s per worker."""

    def test_a_task_runs_without_building_any_cache(self):
        tasks = PopulationFactory.get_nhanes_draw_tasks(
            2, n=20, year=1999, personFilters=None, nhanesWeights=True, distributions=True
        )
        for c in self.CACHES:
            setattr(PopulationFactory, c, None)
        refuse = mock.Mock(side_effect=AssertionError("a worker rebuilt a cache"))
        with (
            mock.patch.object(PopulationFactory, "get_nhanesDf", refuse),
            mock.patch.object(PopulationFactory, "get_nhanesDf_resampled", refuse),
        ):
            people, drawn, accepted = PopulationFactory.draw_people_worker(tasks[0])
        self.assertEqual(10, len(people))
        self.assertGreaterEqual(drawn, accepted)

    def test_the_cached_distributions_are_not_mutated(self):
        before = copy.deepcopy(PopulationFactory.get_crude_distributions())
        PopulationFactory.get_nhanes_people(
            n=40, year=1999, nhanesWeights=True, distributions=True, nWorkers=2
        )
        tasks = PopulationFactory.get_nhanes_draw_tasks(
            2, n=20, year=1999, personFilters=None, nhanesWeights=True, distributions=True
        )
        PopulationFactory.draw_people_worker(tasks[0])
        np.testing.assert_equal(before, PopulationFactory.get_crude_distributions())

    def test_spawned_build_is_not_slowed_by_cache_rebuilds(self):
        PopulationFactory.get_crude_distributions()  # warm the parent
        start = time.perf_counter()
        with _start_method("spawn"):
            PopulationFactory.get_nhanes_people(
                n=200, year=1999, nhanesWeights=True, distributions=True, nWorkers=2
            )
        # a worker rebuilding the NHANES df alone takes ~14 s
        self.assertLess(time.perf_counter() - start, 10)


class TestPeopleReturned(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 400
        cls.pop = PopulationFactory.get_nhanes_population(
            n=cls.n, year=1999, nhanesWeights=True, nWorkers=2
        )
        cls.people = cls.pop._people

    def test_a_series_of_n_people_on_a_range_index(self):
        self.assertIsInstance(self.people, pd.Series)
        self.assertEqual(self.n, len(self.people))
        self.assertIsInstance(self.people.index, pd.RangeIndex)
        self.assertTrue(all(isinstance(p, Person) for p in self.people))

    def test_index_set_once_over_all_chunks(self):
        self.assertEqual(list(range(self.n)), [p._index for p in self.people])

    def test_names_are_nhanes_rows(self):
        names = set(PopulationFactory.get_nhanesDf().name)
        self.assertTrue(all(p._name in names for p in self.people))

    def test_prevalent_outcomes_are_seeded(self):
        # ~3-4% each in NHANES 1999 adults, so none in 400 would mean nothing was seeded
        for outcomeType in (OutcomeType.STROKE, OutcomeType.MI):
            self.assertGreater(
                sum(p.has_outcome_prior_to_simulation(outcomeType) for p in self.people), 0
            )

    def test_population_advances_serially_and_in_parallel(self):
        pop = PopulationFactory.get_nhanes_population(
            n=40, year=1999, nhanesWeights=True, nWorkers=2
        )
        pop.advance(2)
        pop.advance(2, nWorkers=2)
        self.assertEqual(3, pop._waveCompleted)


class TestSplitDraws(unittest.TestCase):
    def test_shares_sum_to_the_totals(self):
        for n, maxDraws, nWorkers in [(10, 1000, 3), (7, 500, 7), (1000, 100000, 4), (5, 9, 2)]:
            shares = PopulationFactory.split_draws(n, maxDraws, nWorkers)
            self.assertEqual(n, sum(s[0] for s in shares))
            self.assertEqual(maxDraws, sum(s[1] for s in shares))
            self.assertTrue(all(s[0] >= 1 and s[1] >= 1 for s in shares))

    def test_workers_are_capped_at_n(self):
        self.assertEqual(3, len(PopulationFactory.split_draws(3, 500, 8)))

    def test_default_budget_is_for_the_total(self):
        n = 50
        shares = PopulationFactory.split_draws(n, None, 4)
        self.assertEqual(max(100 * n, 500), sum(s[1] for s in shares))


class TestBudgetIsReportedOnTheTotal(unittest.TestCase):
    def test_nobody_accepted_reports_every_draw(self):
        pf = _adults_with_person_filter("nobody", lambda x: False)
        with self.assertRaises(RuntimeError) as cm:
            PopulationFactory.get_nhanes_people(
                n=4, year=1999, personFilters=pf, maxDraws=40, nWorkers=2
            )
        self.assertIn("None of the 40 rows", str(cm.exception))

    def test_budget_exhausted_reports_the_aggregate(self):
        # ~30% of adults are 60+, so 60 draws cannot reach 50 people
        pf = _adults_with_person_filter("atLeast60", lambda x: x._age[0] >= 60)
        with self.assertRaises(RuntimeError) as cm:
            PopulationFactory.get_nhanes_people(
                n=50, year=1999, personFilters=pf, maxDraws=60, nWorkers=2
            )
        self.assertIn("n=50", str(cm.exception))
        self.assertIn("in 60 draws", str(cm.exception))
        self.assertIn("Raise maxDraws", str(cm.exception))

    def test_a_failing_worker_raises_in_the_caller_without_hanging(self):
        result = _run_script(
            [
                "from microsim.person.person_filter_factory import PersonFilterFactory",
                "from microsim.population.population_factory import PopulationFactory",
                "def boom(person):",
                "    raise ValueError('boom in worker')",
                "pf = PersonFilterFactory.get_person_filter(['adult'])",
                "pf.add_filter('person', 'boom', boom)",
                "PopulationFactory.get_nhanes_people(n=20, year=1999, personFilters=pf, "
                "nWorkers=2)",
            ],
            timeout=300,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("boom in worker", result.stderr)

    def test_draw_estimate_is_printed_once(self):
        # ~12% of adults are 75+, under the 25% that triggers the estimate
        result = _run_script(
            [
                "from microsim.person.person_filter_factory import PersonFilterFactory",
                "from microsim.population.population_factory import PopulationFactory",
                "pf = PersonFilterFactory.get_person_filter(['adult'])",
                "pf.add_filter('person', 'atLeast75', lambda x: x._age[0] >= 75)",
                "PopulationFactory.get_nhanes_people(n=40, year=1999, personFilters=pf, "
                "nWorkers=2)",
            ]
        )
        self.assertEqual(0, result.returncode, result.stderr)
        estimates = [
            line
            for line in result.stdout.splitlines()
            if line.startswith(("Warning: personFilters accepted", "Warning: none of the"))
        ]
        self.assertEqual(1, len(estimates), result.stdout)


class TestFiltersAndWeightsReachTheWorkers(unittest.TestCase):
    """Every PersonFilter function is a lambda, which only cloudpickle can send to a worker."""

    def test_zero_weight_rows_are_never_drawn(self):
        nhanesDf = PopulationFactory.get_nhanesDf()
        customWeights = (nhanesDf.gender == NHANESGender.MALE.value).astype(float)
        pf = _adults_with_person_filter("atLeast60", lambda x: x._age[0] >= 60)
        people = PopulationFactory.get_nhanes_people(
            n=30, year=1999, personFilters=pf, customWeights=customWeights, nWorkers=2
        )
        self.assertEqual(30, len(people))
        self.assertTrue(all(p._gender == NHANESGender.MALE for p in people))

    def test_a_lambda_with_a_captured_variable_works_in_spawned_workers(self):
        threshold = 65
        pf = _adults_with_person_filter("old", lambda x: x._age[0] >= threshold)
        with _start_method("spawn"):
            people = PopulationFactory.get_nhanes_people(
                n=20, year=1999, personFilters=pf, nWorkers=2
            )
        self.assertEqual(20, len(people))
        self.assertTrue(all(p._age[0] >= threshold for p in people))

    def test_df_filters_hold_for_the_draws(self):
        pf = PersonFilterFactory.get_person_filter(["adult", "lowSBPLimit"])
        pf.add_filter("df", "under60", lambda x: x[DynamicRiskFactorsType.AGE.value] < 60)
        with _start_method("spawn"):
            people = PopulationFactory.get_nhanes_people(
                n=40, year=1999, personFilters=pf, distributions=True, nWorkers=2
            )
        self.assertEqual(40, len(people))
        self.assertGreater(min(p._sbp[0] for p in people), 126)
        self.assertLessEqual(max(p._age[0] for p in people), 59)
        self.assertGreaterEqual(min(p._age[0] for p in people), 18)


class TestParallelMatchesSerial(unittest.TestCase):
    """Two independent weighted draws, so the tolerance is on the difference of their means. 3.5
    standard errors keeps the 6 comparisons at ~0.3% false failures together."""

    @classmethod
    def setUpClass(cls):
        args = dict(n=2000, year=1999, nhanesWeights=True)
        cls.serial = PopulationFactory.get_nhanes_population(**args)._people
        cls.parallel = PopulationFactory.get_nhanes_population(**args, nWorkers=2)._people

    def _assert_close(self, f):
        a = np.array([f(p) for p in self.serial], dtype=float)
        b = np.array([f(p) for p in self.parallel], dtype=float)
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        self.assertAlmostEqual(a.mean(), b.mean(), delta=3.5 * se)

    def test_risk_factors(self):
        for v in ("_age", "_sbp", "_bmi"):
            with self.subTest(v):
                self._assert_close(lambda p: getattr(p, v)[0])

    def test_gender(self):
        self._assert_close(lambda p: p._gender == NHANESGender.FEMALE)

    def test_prevalent_outcomes(self):
        for outcomeType in (OutcomeType.STROKE, OutcomeType.MI):
            with self.subTest(outcomeType):
                self._assert_close(lambda p: p.has_outcome_prior_to_simulation(outcomeType))


class TestArgumentsAreChecked(unittest.TestCase):
    def test_bad_nworkers_is_refused(self):
        for bad in (0, -1, 1.5, True, "2", None):
            with self.subTest(bad), self.assertRaises(RuntimeError):
                PopulationFactory.check_nhanes_people_arguments(n=10, year=1999, nWorkers=bad)

    def test_parallel_without_n_is_refused(self):
        with self.assertRaises(RuntimeError):
            PopulationFactory.check_nhanes_people_arguments(n=None, year=1999, nWorkers=2)

    def test_kaiser_and_state_refuse_parallel(self):
        for popType in (PopulationType.KAISER, PopulationType.STATE):
            with self.subTest(popType), self.assertRaises(RuntimeError) as cm:
                PopulationFactory.get_people(popType, nWorkers=2)
            self.assertIn("nWorkers", str(cm.exception))

    def test_checked_before_any_work(self):
        with (
            mock.patch.object(
                PopulationFactory, "get_nhanesDf", side_effect=AssertionError("work started")
            ),
            self.assertRaises(RuntimeError),
        ):
            PopulationFactory.get_nhanes_people(n=10, year=1999, nWorkers=0)


class TestSerialPathIsUnchanged(unittest.TestCase):
    def test_one_worker_creates_no_pool(self):
        refuse = mock.Mock(side_effect=AssertionError("a pool was created"))
        with (
            mock.patch("multiprocessing.get_context", refuse),
            mock.patch("multiprocessing.Pool", refuse),
        ):
            people = PopulationFactory.get_nhanes_people(n=20, year=1999, nWorkers=1)
        self.assertEqual(20, len(people))


class TestTrials(unittest.TestCase):
    def test_nworkers_reaches_the_nhanes_people_args_only(self):
        self.assertEqual(
            2, NhanesTrialDescription(sampleSize=10, nWorkers=2).peopleArgs["nWorkers"]
        )
        self.assertNotIn("nWorkers", KaiserTrialDescription(sampleSize=10, nWorkers=2).peopleArgs)

    def test_trial_arms_have_unique_indexes(self):
        for trialType in (TrialType.NON_RANDOMIZED, TrialType.POTENTIAL_OUTCOMES):
            with self.subTest(trialType):
                trial = Trial(
                    NhanesTrialDescription(
                        trialType=trialType, sampleSize=30, duration=1, nWorkers=2
                    ),
                    notify=False,
                )
                people = pd.concat([trial.treatedPop._people, trial.controlPop._people])
                self.assertEqual(60, len(people))
                self.assertEqual(60, len({p._index for p in people}))

    def test_trial_runs_end_to_end(self):
        trial = TrialFactory.run_nhanes(
            sampleSize=50,
            duration=2,
            treatmentStrategies="1bpMedsAdded",
            nWorkers=2,
            notify=False,
        )
        self.assertTrue(trial.completed)
        self.assertTrue(trial.analyzed)
        self.assertEqual(100, len(trial.treatedPop._people) + len(trial.controlPop._people))


if __name__ == "__main__":
    unittest.main()
