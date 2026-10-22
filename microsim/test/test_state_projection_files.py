"""Checks that every state projection CSV is well formed, so a malformed file is caught here and
fixed rather than built into a population, and that a year or state without one is refused."""

import glob
import os
import unittest

import pandas as pd

from microsim.common.data_loader import get_absolute_datafile_path
from microsim.population.population_factory import PopulationFactory
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.validation import Validation

CELL = [
    "ageGroup",
    "raceEthnicity",
    "gender",
    "education",
    "smokingStatus",
    "anyPhysicalActivity",
]

CODES = {
    "ageGroup": set(range(1, PopulationFactory.topAgeGroup + 1)),
    # NHANES holds no ASIAN people, so the state pipeline cannot build them (see get_stateDf)
    "raceEthnicity": {int(r) for r in RaceEthnicity if r != RaceEthnicity.ASIAN},
    "gender": {int(g) for g in NHANESGender},
    "education": {int(e) for e in Education},
    "smokingStatus": {int(s) for s in SmokingStatus},
    "anyPhysicalActivity": {0, 1},
}


class TestStateProjectionFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = sorted(glob.glob(get_absolute_datafile_path("state") + "/*.csv"))
        cls.projections = {os.path.basename(path): pd.read_csv(path) for path in paths}

    def test_files_exist(self):
        self.assertGreater(len(self.projections), 0)

    def test_file_names(self):
        for name in self.projections:
            with self.subTest(file=name):
                self.assertRegex(name, r"^pop_projection_[a-z]{2}_\d{4}\.csv$")

    def test_columns(self):
        for name, df in self.projections.items():
            with self.subTest(file=name):
                self.assertEqual(sorted(df.columns), sorted(["state", *CELL, "n"]))

    def test_state_column_matches_file_name(self):
        for name, df in self.projections.items():
            with self.subTest(file=name):
                self.assertEqual(set(df["state"]), {name.split("_")[2].upper()})

    def test_no_missing_values(self):
        for name, df in self.projections.items():
            with self.subTest(file=name):
                self.assertFalse(df.isna().any().any())

    def test_one_row_per_cell(self):
        for name, df in self.projections.items():
            with self.subTest(file=name):
                duplicates = df[df.duplicated(CELL, keep=False)]
                self.assertTrue(duplicates.empty, f"cells with more than one row:\n{duplicates}")

    def test_codes_are_valid(self):
        for name, df in self.projections.items():
            for column, codes in CODES.items():
                with self.subTest(file=name, column=column):
                    self.assertEqual(set(df[column].unique().tolist()) - codes, set())

    def test_counts_are_non_negative_integers(self):
        for name, df in self.projections.items():
            with self.subTest(file=name):
                self.assertTrue(pd.api.types.is_integer_dtype(df["n"]))
                self.assertGreaterEqual(df["n"].min(), 0)


class TestStateProjectionLookup(unittest.TestCase):
    def test_states_of_a_year(self):
        states = PopulationFactory.get_state_projection_states(2025)
        self.assertEqual(len(states), 51)
        self.assertIn("DC", states)

    def test_path_of_a_state_and_year(self):
        path = PopulationFactory.get_state_projection_path(2040, "oh")
        self.assertEqual(os.path.basename(path), "pop_projection_oh_2040.csv")
        self.assertTrue(os.path.isfile(path))

    def test_unknown_year_lists_the_years(self):
        with self.assertRaisesRegex(ValueError, r"2045.*'2025', '2030', '2035', '2040'"):
            PopulationFactory.get_state_projection_path(2045, "OH")

    def test_unknown_state_lists_the_states(self):
        with self.assertRaisesRegex(ValueError, r"ZZ.*'AK'"):
            PopulationFactory.get_state_projection_path(2025, "ZZ")

    def test_population_refuses_unknown_year(self):
        with self.assertRaises(ValueError):
            PopulationFactory.get_state_population(year=2045, state="OH")

    def test_validation_refuses_unknown_year(self):
        # states=None used to find no files, build nothing and die with a StopIteration
        with self.assertRaises(ValueError):
            Validation.states_baseline_pop(year=2045, states=None)


if __name__ == "__main__":
    unittest.main()
