"""Tests for the head counts of state populations (get_dataframe_with_categoricals).

Rows hold fractional expected counts that are rounded once, together, by systematic rounding, so
the total and every projection cell must come within 1 person of n*samplingRate. WY is the
smallest-cell case the old per-row rounding failed on (4,784 people instead of 6,363 at 0.01)."""

import unittest

import numpy as np
import pandas as pd

from microsim.common.data_loader import get_absolute_datafile_path
from microsim.population.population_factory import PopulationFactory
from microsim.validation import Validation

CELL = [
    "ageGroup",
    "raceEthnicity",
    "gender",
    "education",
    "smokingStatus",
    "anyPhysicalActivity",
]


class TestStatePopulationCounts(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        projection = pd.read_csv(
            get_absolute_datafile_path("state") + "/pop_projection_wy_2025.csv"
        )
        self.nForCell = projection.groupby(CELL)["n"].sum()

    def counts_for_cell(self, samplingRate):
        df = PopulationFactory.get_dataframe_with_categoricals(
            year=2025, state="WY", samplingRate=samplingRate
        )
        return df.groupby(CELL).size().reindex(self.nForCell.index, fill_value=0)

    def test_total_is_within_one_person(self):
        counts = self.counts_for_cell(0.01)
        self.assertLessEqual(abs(counts.sum() - self.nForCell.sum() * 0.01), 1)

    def test_every_cell_is_within_one_person(self):
        counts = self.counts_for_cell(0.01)
        self.assertLessEqual((counts - self.nForCell * 0.01).abs().max(), 1)

    def test_full_rate_keeps_the_whole_state(self):
        counts = self.counts_for_cell(1.0)
        self.assertLessEqual(abs(counts.sum() - self.nForCell.sum()), 1)

    def test_every_gender_race_age_has_a_pooled_fit(self):
        keys = PopulationFactory.get_crude_distributions()["mean"].keys()
        for gender in [1, 2]:
            for raceEthnicity in [1, 2, 3, 4, 5]:
                for age in range(82):
                    self.assertIn((gender, raceEthnicity, age), keys)

    def test_state_with_groups_nhanes_lacks_draws(self):
        # IA holds men of race other with less than high school aged 25-29, whom NHANES lacks
        df = PopulationFactory.get_dataframe_with_categoricals(
            year=2025, state="IA", samplingRate=0.01
        )
        df = PopulationFactory.append_dataframe_with_continuous(
            df, PopulationFactory.get_crude_distributions()
        )
        drawn = PopulationFactory.continuous_variables_drawn()
        self.assertFalse(df[drawn].isna().any().any())

    def test_built_population_matches_projection(self):
        # the built Persons, not just the dataframe they are built from
        results = Validation.state_baseline_pop("WY", samplingRate=0.005, verbose=False)
        self.assertLessEqual(abs(results["n"] - results["expectedN"]), 1)
        self.assertLessEqual(results["maxCellError"], 1)
        # not bounded by the rounding like the cells: across 500 seeds at this size the worst
        # was 0.46 pp (smokingStatus), under the ~0.7 pp sd of a simple random sample
        for marginal in results["marginals"].values():
            self.assertLess(marginal["difference"].abs().max(), 1.0)


if __name__ == "__main__":
    unittest.main()
