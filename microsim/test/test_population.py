from microsim.person import Person
from microsim.gcp_model import GCPModel
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.population_factory import PopulationFactory

import unittest
import pandas as pd
import numpy as np
from microsim.outcome import OutcomeType, Outcome
from microsim.population_factory import PopulationFactory

class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.test_n = 50000
        full_nhanes = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        test_nhanes = full_nhanes.loc[full_nhanes.year == 2015]
        ageMeanList = list()
        for i in range(10):
            test_sample = test_nhanes.sample( self.test_n, weights=test_nhanes.WTINT2YR, replace=True )
            ageMeanList += [test_sample.age.mean()]
        self.mean = np.mean(ageMeanList)
        self.sd = np.std(ageMeanList)

    def test_people_from_population(self):
        test_pop = PopulationFactory.get_nhanes_population(n=self.test_n, year=2015, personFilters=None, nhanesWeights=True, distributions=False)
        test_people = test_pop._people

        test_ages = [x._age[0] for x in test_people]
        self.assertTrue(np.mean(test_ages) < self.mean + 2*self.sd)
        self.assertTrue(np.mean(test_ages) > self.mean - 2*self.sd)

def initializeAFib(person):
    return False


class TestPopulationAdvanceOutcomes(unittest.TestCase):
    def setUp(self):
        self.pop = PopulationFactory.get_nhanes_population(n=100, year=1999, personFilters=None, nhanesWeights=True, distributions=False)
        self.pop.advance(1)

    def test_dont_advance_dead_people_in_population(self):

        for person in self.pop._people:
            if len(person._outcomes[OutcomeType.DEATH])==0:
                person._outcomes[OutcomeType.DEATH] = [(person._age[-1], Outcome(OutcomeType.DEATH, True))]

        #if all persons are dead, no risk factor should be predicted for year 2
        self.pop.advance(1)

        expected_risk_factor_length = 1

        for person in self.pop._people:
            self.assertEqual(expected_risk_factor_length, len(person._sbp))


if __name__ == "__main__":
    unittest.main()
