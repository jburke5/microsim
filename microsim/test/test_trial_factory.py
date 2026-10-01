import os
import tempfile
import unittest

from microsim.trials.trial import Trial
from microsim.trials.trial_factory import TrialFactory
from microsim.trials.trial_outcome_assessor import AnalysisType


class TestTrialFactoryNhanes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trial = TrialFactory.run_nhanes(sampleSize=50,
                                            duration=2,
                                            treatmentStrategies="1bpMedsAdded",
                                            year=1999,
                                            nhanesWeights=True,
                                            notify=False)

    def test_returns_trial(self):
        self.assertIsInstance(self.trial, Trial)

    def test_trial_is_completed(self):
        self.assertTrue(self.trial.completed)

    def test_trial_is_analyzed(self):
        self.assertTrue(self.trial.analyzed)

    def test_results_non_empty(self):
        self.assertGreater(len(self.trial.results), 0)

    def test_results_contain_default_analyses(self):
        for analysisType in (AnalysisType.LOGISTIC, AnalysisType.LINEAR,
                             AnalysisType.COX, AnalysisType.RELATIVE_RISK,
                             AnalysisType.INCIDENCE_RATE):
            self.assertIn(analysisType.value, self.trial.results)

    def test_populations_have_expected_size(self):
        self.assertEqual(len(self.trial.treatedPop._people) +
                         len(self.trial.controlPop._people), 100)

    def test_export_results_one_row_per_assessment(self):
        nAssessments = sum(len(byName) for byName in self.trial.results.values())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            self.trial.export_results(path)
            lines = open(path).read().splitlines()
        #4 description lines, then per analysis type: blank, analysis, header, one line per assessment
        self.assertEqual(4 + 3 * len(self.trial.results) + nAssessments, len(lines))


class TestTrialFactoryKaiser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trial = TrialFactory.run_kaiser(sampleSize=50,
                                            duration=2,
                                            treatmentStrategies="1bpMedsAdded",
                                            notify=False)

    def test_returns_trial(self):
        self.assertIsInstance(self.trial, Trial)

    def test_trial_is_completed(self):
        self.assertTrue(self.trial.completed)

    def test_trial_is_analyzed(self):
        self.assertTrue(self.trial.analyzed)

    def test_results_non_empty(self):
        self.assertGreater(len(self.trial.results), 0)

    def test_results_contain_default_analyses(self):
        for analysisType in (AnalysisType.LOGISTIC, AnalysisType.LINEAR,
                             AnalysisType.COX, AnalysisType.RELATIVE_RISK,
                             AnalysisType.INCIDENCE_RATE):
            self.assertIn(analysisType.value, self.trial.results)

    def test_populations_have_expected_size(self):
        self.assertEqual(len(self.trial.treatedPop._people) +
                         len(self.trial.controlPop._people), 100)


if __name__ == "__main__":
    unittest.main()
