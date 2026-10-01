import unittest
import pandas as pd

from microsim.outcomes.outcome import OutcomeType, Outcome, EventOutcomeType
from microsim.population.population import Population
from microsim.population.population_factory import PopulationFactory


class TestGetOutcomeFlagsPerWave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pop = PopulationFactory.get_nhanes_population(
            n=10, year=1999, personFilters=None, nhanesWeights=True, distributions=False
        )
        cls._pop.advance(3)

    def setUp(self):
        # reset all event outcomes on all people so each test starts from a known state
        for person in self._pop._people:
            for eot in EventOutcomeType:
                ot = OutcomeType(eot.value)
                person._outcomes[ot] = []

    def test_returns_one_key_per_event_outcome_type(self):
        person = self._pop._people.iloc[0]
        flags = Population.get_outcome_flags_per_wave(person)
        self.assertEqual(len(flags), len(EventOutcomeType))
        for eot in EventOutcomeType:
            self.assertIn(OutcomeType(eot.value), flags)

    def test_flags_length_matches_waves(self):
        for person in self._pop._people:
            flags = Population.get_outcome_flags_per_wave(person)
            expectedLength = person._waveCompleted + 1
            for outcomeFlags in flags.values():
                self.assertEqual(len(outcomeFlags), expectedLength)

    def test_no_outcomes_returns_all_zeros(self):
        for person in self._pop._people:
            flags = Population.get_outcome_flags_per_wave(person)
            for outcomeFlags in flags.values():
                self.assertTrue(all(f == 0 for f in outcomeFlags))

    def test_flags_are_zero_or_one(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[1], Outcome(OutcomeType.STROKE, False))
        )
        flags = Population.get_outcome_flags_per_wave(person)
        for outcomeFlags in flags.values():
            for f in outcomeFlags:
                self.assertIn(f, [0, 1])

    def test_single_stroke_at_wave_1(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[1], Outcome(OutcomeType.STROKE, False))
        )
        flags = Population.get_outcome_flags_per_wave(person)
        strokeFlags = flags[OutcomeType.STROKE]
        self.assertEqual(strokeFlags[0], 0)
        self.assertEqual(strokeFlags[1], 1)
        for f in strokeFlags[2:]:
            self.assertEqual(f, 0)

    def test_multiple_outcomes_at_different_waves(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[1], Outcome(OutcomeType.STROKE, False))
        )
        person._outcomes[OutcomeType.MI].append((person._age[2], Outcome(OutcomeType.MI, False)))
        flags = Population.get_outcome_flags_per_wave(person)
        self.assertEqual(flags[OutcomeType.STROKE][1], 1)
        self.assertEqual(flags[OutcomeType.STROKE][2], 0)
        self.assertEqual(flags[OutcomeType.MI][1], 0)
        self.assertEqual(flags[OutcomeType.MI][2], 1)

    def test_prior_to_sim_outcomes_excluded(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[0], Outcome(OutcomeType.STROKE, False, priorToSim=True))
        )
        flags = Population.get_outcome_flags_per_wave(person)
        self.assertTrue(all(f == 0 for f in flags[OutcomeType.STROKE]))

    def test_in_sim_outcome_at_same_age_as_prior_to_sim(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[0], Outcome(OutcomeType.STROKE, False, priorToSim=True))
        )
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[2], Outcome(OutcomeType.STROKE, False))
        )
        flags = Population.get_outcome_flags_per_wave(person)
        self.assertEqual(flags[OutcomeType.STROKE][0], 0)
        self.assertEqual(flags[OutcomeType.STROKE][2], 1)

    def test_outcome_type_not_in_person_outcomes(self):
        """If a person's _outcomes dict doesn't have a key for an EventOutcomeType, it should be
        all zeros."""
        person = self._pop._people.iloc[0]
        if OutcomeType.STROKE in person._outcomes:
            del person._outcomes[OutcomeType.STROKE]
        flags = Population.get_outcome_flags_per_wave(person)
        self.assertTrue(all(f == 0 for f in flags[OutcomeType.STROKE]))


class TestGetOutcomeHistoryPerWave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pop = PopulationFactory.get_nhanes_population(
            n=10, year=1999, personFilters=None, nhanesWeights=True, distributions=False
        )
        cls._pop.advance(5)

    def setUp(self):
        for person in self._pop._people:
            for eot in EventOutcomeType:
                ot = OutcomeType(eot.value)
                person._outcomes[ot] = []

    def test_returns_one_key_per_event_outcome_type(self):
        person = self._pop._people.iloc[0]
        history = Population.get_outcome_history_per_wave(person)
        self.assertEqual(len(history), len(EventOutcomeType))
        for eot in EventOutcomeType:
            self.assertIn(OutcomeType(eot.value), history)

    def test_no_outcomes_returns_all_zeros(self):
        for person in self._pop._people:
            history = Population.get_outcome_history_per_wave(person)
            for historyFlags in history.values():
                self.assertTrue(all(f == 0 for f in historyFlags))

    def test_stroke_at_wave_1_persists_from_wave_2(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[1], Outcome(OutcomeType.STROKE, False))
        )
        history = Population.get_outcome_history_per_wave(person)
        strokeHistory = history[OutcomeType.STROKE]
        self.assertEqual(strokeHistory[0], 0)
        self.assertEqual(strokeHistory[1], 0)
        for f in strokeHistory[2:]:
            self.assertEqual(f, 1)

    def test_stroke_at_wave_2_zero_before(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[2], Outcome(OutcomeType.STROKE, False))
        )
        history = Population.get_outcome_history_per_wave(person)
        strokeHistory = history[OutcomeType.STROKE]
        self.assertEqual(strokeHistory[0], 0)
        self.assertEqual(strokeHistory[1], 0)
        self.assertEqual(strokeHistory[2], 0)
        for f in strokeHistory[3:]:
            self.assertEqual(f, 1)

    def test_multiple_outcomes_independent_histories(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[1], Outcome(OutcomeType.STROKE, False))
        )
        person._outcomes[OutcomeType.MI].append((person._age[3], Outcome(OutcomeType.MI, False)))
        history = Population.get_outcome_history_per_wave(person)
        # stroke at wave 1: history is 0 at waves 0-1, 1 from wave 2 onward
        self.assertEqual(history[OutcomeType.STROKE][0], 0)
        self.assertEqual(history[OutcomeType.STROKE][1], 0)
        self.assertEqual(history[OutcomeType.STROKE][2], 1)
        self.assertEqual(history[OutcomeType.STROKE][3], 1)
        # mi at wave 3: history is 0 at waves 0-3, 1 from wave 4 onward
        self.assertEqual(history[OutcomeType.MI][0], 0)
        self.assertEqual(history[OutcomeType.MI][2], 0)
        self.assertEqual(history[OutcomeType.MI][3], 0)
        self.assertEqual(history[OutcomeType.MI][4], 1)

    def test_prior_to_sim_outcomes_excluded(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[0], Outcome(OutcomeType.STROKE, False, priorToSim=True))
        )
        history = Population.get_outcome_history_per_wave(person)
        self.assertTrue(all(f == 0 for f in history[OutcomeType.STROKE]))

    def test_history_values_are_zero_or_one(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[1], Outcome(OutcomeType.STROKE, False))
        )
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[3], Outcome(OutcomeType.STROKE, False))
        )
        history = Population.get_outcome_history_per_wave(person)
        for historyFlags in history.values():
            for f in historyFlags:
                self.assertIn(f, [0, 1])


class TestGetAllPersonYearsAsDf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pop = PopulationFactory.get_nhanes_population(
            n=10, year=1999, personFilters=None, nhanesWeights=True, distributions=False
        )
        cls._pop.advance(2)

    def setUp(self):
        for person in self._pop._people:
            for eot in EventOutcomeType:
                ot = OutcomeType(eot.value)
                person._outcomes[ot] = []

    def test_df_has_outcome_columns(self):
        df = self._pop.get_all_person_years_as_df()
        for eot in EventOutcomeType:
            self.assertIn(eot.value, df.columns)

    def test_df_has_outcome_history_columns(self):
        df = self._pop.get_all_person_years_as_df()
        for eot in EventOutcomeType:
            self.assertIn(eot.value + "History", df.columns)

    def test_df_still_has_risk_factor_columns(self):
        df = self._pop.get_all_person_years_as_df()
        self.assertIn("name", df.columns)
        self.assertIn("age", df.columns)
        self.assertIn("sbp", df.columns)

    def test_df_row_count(self):
        df = self._pop.get_all_person_years_as_df()
        expectedRows = sum(person._waveCompleted + 1 for person in self._pop._people)
        self.assertEqual(len(df), expectedRows)

    def test_df_no_outcomes_all_zeros(self):
        df = self._pop.get_all_person_years_as_df()
        for eot in EventOutcomeType:
            self.assertEqual(df[eot.value].sum(), 0)

    def test_df_with_manually_set_stroke(self):
        person = self._pop._people.iloc[0]
        person._outcomes[OutcomeType.STROKE].append(
            (person._age[1], Outcome(OutcomeType.STROKE, False))
        )
        df = self._pop.get_all_person_years_as_df()
        self.assertGreater(df["stroke"].sum(), 0)
        self.assertEqual(df["mi"].sum(), 0)

    def test_df_outcome_columns_are_numeric(self):
        df = self._pop.get_all_person_years_as_df()
        for eot in EventOutcomeType:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[eot.value]))


if __name__ == "__main__":
    unittest.main()
