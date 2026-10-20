"""Checks that every state projection CSV is well formed, so a malformed file is caught here and
fixed rather than built into a population."""

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


if __name__ == "__main__":
    unittest.main()
