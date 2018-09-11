from mcm.population import NHANESDirectSamplePopulation
import unittest
import pandas as pd
import numpy as np


class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.test_n = 10000
        self.pandas_seed = 78483
        test_nhanes = pd.read_stata("mcm/nhanes2015-2016Combined.dta")
        self.test_sample = test_nhanes.sample(
            self.test_n, weights=test_nhanes.wtint2yr, random_state=self.pandas_seed, replace=True)

    def test_people_from_population(self):
        test_population = NHANESDirectSamplePopulation(
            self.test_n, self.pandas_seed)
        test_people = test_population._people

        test_ages = [x._age[0] for x in test_people]
        self.assertAlmostEqual(
            np.mean(test_ages), self.test_sample.age.mean(), delta=0.000001)


if __name__ == "__main__":
    unittest.main()
