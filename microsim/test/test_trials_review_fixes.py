import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.trials.cox_regression_analysis import CoxRegressionAnalysis
from microsim.trials.incidence_rate_analysis import IncidenceRateAnalysis
from microsim.trials.linear_regression_analysis import LinearRegressionAnalysis
from microsim.trials.logistic_regression_analysis import LogisticRegressionAnalysis
from microsim.trials.regression_analysis import RegressionAnalysis
from microsim.trials.relative_risk_analysis import RelativeRiskAnalysis
from microsim.trials.trial import Trial
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial_outcome_assessor import AnalysisType, TrialOutcomeAssessor
from microsim.trials.trial_type import TrialType
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType, TreatmentStrategyStatus


class FakePopulation:
    def __init__(self, n, attrs=None):
        self._n = n
        self._attrs = attrs if attrs is not None else {}

    def get_attr(self, attr):
        return self._attrs[attr]


class FakeTrial:
    def __init__(self, treatedPop, controlPop, blockFactors=[]):
        self.treatedPop = treatedPop
        self.controlPop = controlPop
        self.trialDescription = type("FakeDescription", (), {"blockFactors": blockFactors})()


def make_regression_df(rng, n, categorical=True, continuous=False, binaryOutcome=False, cox=False):
    df = pd.DataFrame({"treatment": rng.integers(0, 2, n)})
    if categorical:
        df["raceEthnicity"] = rng.choice([r for r in RaceEthnicity], size=n)
    if continuous:
        df["bmi"] = rng.normal(28, 4, n)
    df["outcome"] = rng.integers(0, 2, n) if binaryOutcome else rng.normal(size=n)
    if cox:
        df["outcomeTime"] = rng.integers(1, 6, n)
    return df


class TestGetTrialOutcomeDf(unittest.TestCase):
    '''get_trial_outcome_df: all block factors become columns, dynamic block factors are rejected.'''

    def test_all_block_factors_become_columns(self):
        treated = FakePopulation(3, {"gender": [1, 2, 1], "raceEthnicity": [3, 4, 3]})
        control = FakePopulation(2, {"gender": [2, 2], "raceEthnicity": [4, 5]})
        trial = FakeTrial(treated, control, blockFactors=["gender", "raceEthnicity"])
        df = RegressionAnalysis().get_trial_outcome_df(trial, {"outcome": lambda p: [0.] * p._n}, "linear")
        self.assertEqual(["treatment", "outcome", "gender", "raceEthnicity"], list(df.columns))
        self.assertEqual([1, 1, 1, 0, 0], list(df["treatment"]))
        self.assertEqual([1, 2, 1, 2, 2], list(df["gender"]))
        self.assertEqual([3, 4, 3, 4, 5], list(df["raceEthnicity"]))

    def test_dynamic_block_factor_raises(self):
        treated = FakePopulation(2, {"age": [[60, 61], [70, 71]]})
        control = FakePopulation(2, {"age": [[50, 51], [55, 56]]})
        trial = FakeTrial(treated, control, blockFactors=["age"])
        with self.assertRaises(RuntimeError):
            RegressionAnalysis().get_trial_outcome_df(trial, {"outcome": lambda p: [0.] * p._n}, "linear")

    def test_is_categorical(self):
        self.assertTrue(RegressionAnalysis.is_categorical("raceEthnicity"))
        self.assertTrue(RegressionAnalysis.is_categorical("gender"))
        self.assertFalse(RegressionAnalysis.is_categorical("bmi"))
        self.assertFalse(RegressionAnalysis.is_categorical("age"))


class TestLinearRegressionAnalysis(unittest.TestCase):
    '''Categorical block factors are dummy-encoded via C(), degenerate fits return NaNs.'''

    def test_categorical_block_factor_matches_manual_dummy_fit(self):
        rng = np.random.default_rng(7)
        df = make_regression_df(rng, 400)
        analysis = LinearRegressionAnalysis()
        analysis.get_trial_outcome_df = lambda *args: df
        trial = FakeTrial(None, None, blockFactors=["raceEthnicity"])
        coef, se, pValue, intercept = analysis.analyze(trial, {}, "linear")
        reference = smf.ols("outcome ~ treatment + C(raceEthnicity)", df).fit()
        self.assertAlmostEqual(reference.params["treatment"], coef, places=12)
        self.assertAlmostEqual(reference.bse["treatment"], se, places=12)

    def test_continuous_block_factor_enters_linearly(self):
        rng = np.random.default_rng(7)
        df = make_regression_df(rng, 400, categorical=False, continuous=True)
        analysis = LinearRegressionAnalysis()
        analysis.get_trial_outcome_df = lambda *args: df
        trial = FakeTrial(None, None, blockFactors=["bmi"])
        coef, se, pValue, intercept = analysis.analyze(trial, {}, "linear")
        reference = smf.ols("outcome ~ treatment + bmi", df).fit()
        self.assertAlmostEqual(reference.params["treatment"], coef, places=12)

    def test_degenerate_fit_returns_nans(self):
        df = pd.DataFrame({"treatment": [0, 1] * 20, "outcome": [np.nan] * 40})
        analysis = LinearRegressionAnalysis()
        analysis.get_trial_outcome_df = lambda *args: df
        trial = FakeTrial(None, None)
        result = analysis.analyze(trial, {}, "linear")
        self.assertTrue(all(math.isnan(r) for r in result))
        self.assertEqual(len(LinearRegressionAnalysis.columns), len(result))


class TestLogisticRegressionAnalysis(unittest.TestCase):
    '''Perfect separation returns NaNs, categorical block factors are dummy-encoded.'''

    def test_perfect_separation_returns_nans(self):
        df = pd.DataFrame({"treatment": [0, 1] * 20})
        df["outcome"] = df["treatment"]
        analysis = LogisticRegressionAnalysis()
        analysis.get_trial_outcome_df = lambda *args: df
        trial = FakeTrial(None, None)
        result = analysis.analyze(trial, {}, "logistic")
        self.assertTrue(all(math.isnan(r) for r in result))
        self.assertEqual(len(LogisticRegressionAnalysis.columns), len(result))

    def test_categorical_block_factor_matches_manual_dummy_fit(self):
        rng = np.random.default_rng(7)
        df = make_regression_df(rng, 400, binaryOutcome=True)
        analysis = LogisticRegressionAnalysis()
        analysis.get_trial_outcome_df = lambda *args: df
        trial = FakeTrial(None, None, blockFactors=["raceEthnicity"])
        coef, se, pValue, intercept = analysis.analyze(trial, {}, "logistic")
        reference = smf.logit("outcome ~ treatment + C(raceEthnicity)", df).fit(disp=False)
        self.assertAlmostEqual(reference.params["treatment"], coef, places=8)


class TestCoxRegressionAnalysis(unittest.TestCase):
    '''Categorical block factors are dummy-encoded, the fitter is fresh per call, 4th element is None.'''

    def test_categorical_block_factor_is_dummy_encoded(self):
        rng = np.random.default_rng(7)
        df = make_regression_df(rng, 400, binaryOutcome=True, cox=True)
        analysis = CoxRegressionAnalysis()
        analysis.get_trial_outcome_df = lambda *args: df
        trial = FakeTrial(None, None, blockFactors=["raceEthnicity"])
        result = analysis.analyze(trial, {}, "cox")
        self.assertEqual(len(CoxRegressionAnalysis.columns), len(result))
        coef, se, pValue, fourth = result
        self.assertTrue(math.isfinite(coef))
        self.assertIsNone(fourth)
        dummyCovariates = [c for c in analysis.cph.params_.index if c.startswith("raceEthnicity_")]
        self.assertGreater(len(dummyCovariates), 1) #a single linear term would be just "raceEthnicity"
        self.assertNotIn("raceEthnicity", analysis.cph.params_.index)

    def test_fresh_fitter_per_analyze_call(self):
        rng = np.random.default_rng(7)
        df = make_regression_df(rng, 100, categorical=False, binaryOutcome=True, cox=True)
        analysis = CoxRegressionAnalysis()
        analysis.get_trial_outcome_df = lambda *args: df
        trial = FakeTrial(None, None)
        analysis.analyze(trial, {}, "cox")
        firstFitter = analysis.cph
        analysis.analyze(trial, {}, "cox")
        self.assertIsNot(firstFitter, analysis.cph)


class TestRelativeRiskAnalysis(unittest.TestCase):
    '''Each arm uses its own denominator, degenerate inputs give NaN instead of crashing.'''

    class Pop:
        def __init__(self, n, count, meds):
            self._n, self._count, self._meds = n, count, meds

        def has_any_meds_added(self):
            return self._meds

    def test_unequal_arms_use_their_own_denominators(self):
        treated = self.Pop(120, 30, [True] * 60 + [False] * 40)
        control = self.Pop(80, 20, [])
        trial = FakeTrial(treated, control)
        result = RelativeRiskAnalysis().analyze(trial, {"outcome": lambda p: p._count}, "relativeRisk")
        self.assertEqual(len(RelativeRiskAnalysis.columns), len(result))
        relativeRisk, tRisk, cRisk = result[0], result[3], result[8]
        self.assertAlmostEqual(30 / 120, tRisk)
        self.assertAlmostEqual(20 / 80, cRisk)
        self.assertAlmostEqual(1.0, relativeRisk)

    def test_zero_denominator_returns_nans_not_crash(self):
        risk, ciLower, ciUpper, ciLowerWilson, ciUpperWilson = RelativeRiskAnalysis().get_absolute_risk(0, 0)
        self.assertTrue(math.isnan(risk))
        self.assertIsNone(ciLower)
        self.assertIsNone(ciUpper)
        self.assertTrue(math.isnan(ciLowerWilson))
        self.assertTrue(math.isnan(ciUpperWilson))

    def test_efficiency_nan_when_no_meds_added(self):
        treated = self.Pop(120, 30, [None] * 10) #None means not in any treatment strategy
        control = self.Pop(120, 20, [])
        trial = FakeTrial(treated, control)
        result = RelativeRiskAnalysis().analyze(trial, {"outcome": lambda p: p._count}, "relativeRisk")
        self.assertTrue(math.isnan(result[16]))


class TestIncidenceRateZeroPersonYears(unittest.TestCase):

    def test_zero_person_years_returns_nan(self):
        class Pop:
            def __init__(self, pairs):
                self._pairs = pairs

            def get_followup_events_and_person_years(self, outcomesTypeList, wave):
                return self._pairs

        trial = FakeTrial(Pop([(True, 3), (False, 5)]), Pop([(False, 0), (False, 0)]))
        result = IncidenceRateAnalysis().analyze(
            trial, {"eventAndTime": lambda p: p.get_followup_events_and_person_years([], 0)}, "incidenceRate")
        self.assertEqual(len(IncidenceRateAnalysis.columns), len(result))
        treatedRate, controlRate = result
        self.assertAlmostEqual(125.0, treatedRate)
        self.assertTrue(math.isnan(controlRate))


class TestTrialDescriptionDefaults(unittest.TestCase):
    '''Mutable-default fixes and single-block-factor validation.'''

    def test_fresh_treatment_strategy_repository_per_description(self):
        d1 = NhanesTrialDescription()
        d2 = NhanesTrialDescription()
        self.assertIsNot(d1.treatmentStrategies, d2.treatmentStrategies)

    def test_block_factors_default_and_copy(self):
        d1 = NhanesTrialDescription()
        self.assertEqual([], d1.blockFactors)
        source = ["gender"]
        d2 = NhanesTrialDescription(trialType=TrialType.COMPLETELY_RANDOMIZED_IN_BLOCKS, blockFactors=source)
        source.append("raceEthnicity")
        self.assertEqual(["gender"], d2.blockFactors)
        d3 = NhanesTrialDescription(trialType=TrialType.COMPLETELY_RANDOMIZED_IN_BLOCKS, blockFactors=("gender",))
        self.assertEqual(["gender"], d3.blockFactors)

    def test_more_than_one_block_factor_raises(self):
        with self.assertRaises(RuntimeError):
            NhanesTrialDescription(trialType=TrialType.COMPLETELY_RANDOMIZED_IN_BLOCKS,
                                   blockFactors=["gender", "raceEthnicity"])


class TestTrialOutcomeAssessorValidation(unittest.TestCase):
    '''Key-set validation and raise-instead-of-print behavior.'''

    def test_unknown_analysis_raises(self):
        with self.assertRaises(RuntimeError):
            TrialOutcomeAssessor().add_outcome_assessment("a", {"outcome": lambda x: x}, "nope")

    def test_duplicate_name_raises(self):
        toa = TrialOutcomeAssessor()
        toa.add_outcome_assessment("a", {"outcome": lambda x: x}, "logistic")
        with self.assertRaises(RuntimeError):
            toa.add_outcome_assessment("a", {"outcome": lambda x: x}, "logistic")

    def test_required_keys_per_analysis(self):
        toa = TrialOutcomeAssessor()
        toa.add_outcome_assessment("ok1", {"outcome": lambda x: x}, "logistic")
        toa.add_outcome_assessment("ok2", {"outcome": lambda x: x, "time": lambda x: x}, "cox")
        toa.add_outcome_assessment("ok3", {"eventAndTime": lambda x: x}, "incidenceRate")
        for name, functionDict, analysis in [("bad1", {"wrongKey": lambda x: x}, "logistic"),
                                             ("bad2", {"outcome": lambda x: x, "tme": lambda x: x}, "cox"),
                                             ("bad3", {"outcome": lambda x: x, "time": lambda x: x}, "incidenceRate")]:
            with self.assertRaises(RuntimeError):
                toa.add_outcome_assessment(name, functionDict, analysis)

    def test_rm_missing_name_raises(self):
        toa = TrialOutcomeAssessor()
        toa.add_outcome_assessment("a", {"outcome": lambda x: x}, "logistic")
        toa.rm_outcome_assessment("a")
        self.assertNotIn("a", toa._assessments)
        with self.assertRaises(RuntimeError):
            toa.rm_outcome_assessment("a")


class TestTrialGuardsAndFormatting(unittest.TestCase):
    '''analyze() guards and __str__ result formatting, without building populations.'''

    def make_bare_trial(self):
        trial = Trial.__new__(Trial)
        trial.trialDescription = NhanesTrialDescription(sampleSize=10, duration=2)
        trial.completed = False
        trial.analyzed = False
        trial.results = dict()
        trial.pythonVersion = "x"
        return trial

    def test_analyze_before_run_raises(self):
        trial = self.make_bare_trial()
        toa = TrialOutcomeAssessor()
        with self.assertRaises(RuntimeError):
            trial.analyze(toa)
        self.assertFalse(trial.analyzed)

    def test_failed_analysis_does_not_mark_analyzed(self):
        trial = self.make_bare_trial()
        trial.completed = True
        trial.treatedPop = FakePopulation(2)
        trial.controlPop = FakePopulation(2)

        def explode(pop):
            raise ValueError("boom")

        toa = TrialOutcomeAssessor()
        toa.add_outcome_assessment("boom", {"outcome": explode}, "linear")
        with self.assertRaises(ValueError):
            trial.analyze(toa)
        self.assertFalse(trial.analyzed)

    def test_str_result_formatting(self):
        trial = self.make_bare_trial()
        trial.completed = True
        trial.analyzed = True
        trial.results = {AnalysisType.LINEAR.value:
                         {"demo": (1.23456, None, float('inf'), float('-inf'), float('nan'))}}
        line = [l for l in str(trial).splitlines() if "demo" in l][0]
        self.assertIn("1.235", line)
        self.assertIn("inf", line)
        self.assertIn("-inf", line)
        self.assertIn("nan", line)

    def test_export_before_analyze_raises(self):
        trial = self.make_bare_trial()
        with self.assertRaises(RuntimeError):
            trial.export_results("x.csv")

    def test_export_results_csv(self):
        trial = self.make_bare_trial()
        trial.completed = True
        trial.analyzed = True
        trial.results = {AnalysisType.INCIDENCE_RATE.value: {"strokeIR": (5.0, 7.0)},
                         AnalysisType.COX.value: {"deathCox": (0.1, 0.2, 0.3, None)},
                         AnalysisType.LINEAR.value: {"demo": (1.23456, float('nan'), float('inf'), float('-inf'))}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            trial.export_results(path)
            lines = open(path).read().splitlines()
        #description once at the top, then one block per analysis type in enum order (not dict order)
        self.assertEqual(["popType,nhanes", "sampleSize,10", "duration,2", "treatmentStrategies,",
                          "", "analysis,linear", "assessment,coef,se,pValue,intercept", "demo,1.235,nan,inf,-inf",
                          "", "analysis,cox", "assessment,coef,se,pValue,intercept", "deathCox,0.100,0.200,0.300,",
                          "", "analysis,incidenceRate", "assessment,treatedRatePer1000PY,controlRatePer1000PY",
                          "strokeIR,5.000,7.000"], lines)

    def test_results_dfs_rejects_wrong_tuple_length(self):
        trial = self.make_bare_trial()
        trial.analyzed = True
        trial.results = {AnalysisType.LINEAR.value: {"demo": (1.0, 2.0)}}
        with self.assertRaises(ValueError):
            trial.get_results_dfs()

    def test_run_analyze_export_default_off(self):
        trial = self.make_bare_trial()
        trial.run = lambda notify=True: setattr(trial, "completed", True)
        trial.treatedPop = FakePopulation(2)
        trial.controlPop = FakePopulation(2)
        toa = TrialOutcomeAssessor()
        with tempfile.TemporaryDirectory() as d:
            trial.run_analyze(toa)
            self.assertEqual([], os.listdir(d))
            trial.run_analyze(toa, exportPath=os.path.join(d, "r.csv"))
            self.assertEqual(["r.csv"], os.listdir(d))


class TestTrialRunIsolationAndRandomization(unittest.TestCase):
    '''run() mutates only the trial's own strategy copy; randomization draws from the description rng.'''

    @classmethod
    def setUpClass(cls):
        cls.description = NhanesTrialDescription(sampleSize=20, duration=2, treatmentStrategies="1bpMedsAdded")
        cls.trial = Trial(cls.description)
        cls.trial.run(notify=False)

    def test_description_strategies_not_mutated_by_run(self):
        bp = TreatmentStrategiesType.BP.value
        self.assertEqual(TreatmentStrategyStatus.BEGIN, self.description.treatmentStrategies._repository[bp].status)
        self.assertEqual(TreatmentStrategyStatus.MAINTAIN, self.trial.treatmentStrategies._repository[bp].status)

    def test_second_trial_from_same_description_starts_at_begin(self):
        bp = TreatmentStrategiesType.BP.value
        secondTrial = Trial(self.description)
        self.assertEqual(TreatmentStrategyStatus.BEGIN, secondTrial.treatmentStrategies._repository[bp].status)

    def test_complete_randomization_uses_description_rng(self):
        people = pd.concat([self.trial.treatedPop._people, self.trial.controlPop._people])
        splits = []
        for _ in range(2):
            self.description._rng = np.random.default_rng(42)
            treated, control = self.trial.randomize_trial_people(people)
            self.assertEqual(len(people) // 2, len(treated))
            splits.append(set(map(lambda p: p._index, treated)))
        self.assertEqual(splits[0], splits[1]) #same seed, same assignment: the description rng drives the split


if __name__ == "__main__":
    unittest.main()
