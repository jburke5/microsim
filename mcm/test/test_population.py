from mcm.population import NHANESDirectSamplePopulation
from mcm.population import Population
from mcm.person import Person
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.education import Education

import unittest
import pandas as pd
import numpy as np


class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.test_n = 10000
        self.pandas_seed = 78483
        full_nhanes = pd.read_stata("mcm/data/fullyImputedDataset.dta")
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


def initializeAFib(person):
    return None


class TestPopulationAdvanceOutcomes(unittest.TestCase):

    def setUp(self):
        self.joe = Person(
            42,
            NHANESGender.MALE,
            NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            140,
            90,
            5.5,
            50,
            200,
            25,
            90,
            150,
            70,
            0,
            Education.COLLEGEGRADUATE,
            SmokingStatus.NEVER,
            initializeAFib,
            selfReportStrokeAge=None,
            selfReportMIAge=None,
            dfIndex=1,
            diedBy2015=0)

    def test_dont_advance_dead_people_in_population(self):
        self.dummy_population = Population([self.joe])
        self.joe._alive.append(False)
        expected_risk_factor_length = len(self.joe._sbp)

        # this should NOT raise an error if it is (correctly) not trying to
        #  advance on poor dead, joe
        self.dummy_population.advance(1)

        self.assertEqual(expected_risk_factor_length, len(self.joe._sbp))


if __name__ == "__main__":
    unittest.main()
