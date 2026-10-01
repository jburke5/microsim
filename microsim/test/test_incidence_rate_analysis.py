import unittest

from microsim.trials.incidence_rate_analysis import IncidenceRateAnalysis
from microsim.trials.trial_outcome_assessor import TrialOutcomeAssessor, AnalysisType
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
from microsim.outcomes.outcome import OutcomeType


class MockPopulation:
    """Mock population for testing incidence rate analysis."""

    def __init__(self, outcomes, person_years):
        """
        Args:
            outcomes: list of booleans indicating who had the event
            person_years: list of integers indicating person-years at risk
        """
        self._outcomes = outcomes
        self._person_years = person_years
        self._n = len(outcomes)
        self._waveCompleted = 4

    def get_followup_events_and_person_years(self, outcomes_list, wave):
        return list(zip(self._outcomes, self._person_years))


class MockTrial:
    """Mock trial for testing incidence rate analysis."""

    def __init__(self, treated_pop, control_pop):
        self.treatedPop = treated_pop
        self.controlPop = control_pop


class TestIncidenceRateAnalysis(unittest.TestCase):

    def test_basic_calculation(self):
        """Test that incidence rate is calculated correctly."""
        # Treated: 2 events in 50 person-years = 40 per 1000 PY
        # Control: 4 events in 50 person-years = 80 per 1000 PY
        treated_pop = MockPopulation(
            outcomes=[True, True, False, False, False],  # 2 events
            person_years=[10, 10, 10, 10, 10]  # 50 person-years
        )
        control_pop = MockPopulation(
            outcomes=[True, True, True, True, False],  # 4 events
            person_years=[10, 10, 10, 10, 10]  # 50 person-years
        )
        trial = MockTrial(treated_pop, control_pop)

        analysis = IncidenceRateAnalysis()
        result = analysis.analyze(
            trial,
            {
                "eventAndTime": lambda x: x.get_followup_events_and_person_years([OutcomeType.STROKE], x._waveCompleted)
            },
            "incidenceRate"
        )

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 40.0)  # treated rate
        self.assertAlmostEqual(result[1], 80.0)  # control rate

    def test_zero_events(self):
        """Test handling of zero events."""
        treated_pop = MockPopulation(
            outcomes=[False, False, False],
            person_years=[10, 10, 10]
        )
        control_pop = MockPopulation(
            outcomes=[False, False, False],
            person_years=[10, 10, 10]
        )
        trial = MockTrial(treated_pop, control_pop)

        analysis = IncidenceRateAnalysis()
        result = analysis.analyze(
            trial,
            {
                "eventAndTime": lambda x: x.get_followup_events_and_person_years([OutcomeType.STROKE], x._waveCompleted)
            },
            "incidenceRate"
        )

        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)

    def test_analysis_type_enum_exists(self):
        """Test that INCIDENCE_RATE is in AnalysisType enum."""
        self.assertEqual(AnalysisType.INCIDENCE_RATE.value, "incidenceRate")

    def test_trial_outcome_assessor_accepts_incidence_rate(self):
        """Test that TrialOutcomeAssessor accepts incidence rate assessments."""
        toa = TrialOutcomeAssessor()
        toa.add_outcome_assessment(
            "testIR",
            {
                "eventAndTime": lambda x: x.get_followup_events_and_person_years([OutcomeType.STROKE], x._waveCompleted)
            },
            AnalysisType.INCIDENCE_RATE.value
        )
        self.assertIn("testIR", toa._assessments)

    def test_factory_includes_default_incidence_rate_assessments(self):
        """Test that factory adds default incidence rate assessments."""
        toa = TrialOutcomeAssessorFactory.get_trial_outcome_assessor(addDefaultAssessments=True)
        self.assertIn("strokeIR", toa._assessments)
        self.assertIn("miIR", toa._assessments)
        self.assertIn("deathIR", toa._assessments)
        self.assertIn("dementiaIR", toa._assessments)


if __name__ == "__main__":
    unittest.main()
