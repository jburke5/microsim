from microsim.population import NHANESDirectSamplePopulation, ClonePopulation
from microsim.person import Person
from microsim.gcp_model import GCPModel
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory

import unittest
import pandas as pd
import numpy as np


class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.test_n = 10000
        self.pandas_seed = 78483
        full_nhanes = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        test_nhanes = full_nhanes.loc[full_nhanes.year == 2015]
        self.test_sample = test_nhanes.sample(
            self.test_n, weights=test_nhanes.WTINT2YR, random_state=self.pandas_seed, replace=True
        )

    def test_people_from_population(self):
        test_population = NHANESDirectSamplePopulation(
            n=self.test_n, year=2015, random_seed=self.pandas_seed
        )
        test_people = test_population._people

        test_ages = [x._age[0] for x in test_people]
        self.assertAlmostEqual(np.mean(test_ages), self.test_sample.age.mean(), delta=0.000001)


def initializeAFib(person):
    return False


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
            AlcoholCategory.NONE,
            0,
            0,
            0,
            0,
            initializeAFib,
            selfReportStrokeAge=None,
            selfReportMIAge=None,
            dfIndex=1,
            diedBy2015=0,
        )

    def test_dont_advance_dead_people_in_population(self):
        # add GCP to advance successfully
        joe_base_gcp = GCPModel().get_risk_for_person(self.joe)
        self.joe._gcp.append(joe_base_gcp)
        # use ClonePopulation: sets up repositories, populationIndex, and 2+ people to workaround
        self.dummy_population = ClonePopulation(self.joe, 2)
        initial_joe = self.dummy_population._people.iloc[0]
        initial_joe._alive.append(False)
        expected_risk_factor_length = len(initial_joe._sbp)

        # this should NOT raise an error if it is (correctly) not trying to
        #  advance on poor dead, joe
        self.dummy_population.advance_vectorized(1)

        advanced_joe = self.dummy_population._people.iloc[0]
        self.assertEqual(expected_risk_factor_length, len(advanced_joe._sbp))


if __name__ == "__main__":
    unittest.main()
