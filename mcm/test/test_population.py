from mcm.population import NHANESDirectSamplePopulation
import unittest
import pandas as pd
import numpy as np


class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.test_n = 10000
        self.pandas_seed = 78483
        full_nhanes = pd.read_stata("mcm/fullyImputedDataset.dta")
        test_nhanes = full_nhanes.loc[full_nhanes.year == 2015]
        self.test_sample = test_nhanes.sample(
            self.test_n, weights=test_nhanes.WTINT2YR, random_state=self.pandas_seed, replace=True)

    def test_people_from_population(self):
        test_population = NHANESDirectSamplePopulation(
            n=self.test_n, year=2015, random_seed=self.pandas_seed)
        test_people = test_population._people

        test_ages = [x._age[0] for x in test_people]
        self.assertAlmostEqual(
            np.mean(test_ages), self.test_sample.age.mean(), delta=0.000001)


if __name__ == "__main__":
    unittest.main()
