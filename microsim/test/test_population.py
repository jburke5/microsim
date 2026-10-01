#from microsim.person.person import Person
#from microsim.outcomes.gcp_model import GCPModel
#from microsim.risk_factors.gender import NHANESGender
#from microsim.risk_factors.race_ethnicity import RaceEthnicity
#from microsim.risk_factors.smoking_status import SmokingStatus
#from microsim.risk_factors.education import Education
#from microsim.risk_factors.alcohol_category import AlcoholCategory
#from microsim.population.population_factory import PopulationFactory

import unittest
import pandas as pd
import numpy as np
from microsim.outcomes.outcome import OutcomeType, Outcome
from microsim.population.population_factory import PopulationFactory
from microsim.common.data_loader import get_absolute_datafile_path

class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.test_n = 50000
        full_nhanes = pd.read_stata(get_absolute_datafile_path("fullyImputedDataset.dta"))
        test_nhanes = full_nhanes.loc[(full_nhanes.year == 2015) & (full_nhanes.age >= 18)]
        ageMeanList = list()
        for i in range(100):
            test_sample = test_nhanes.sample( self.test_n, weights=test_nhanes.WTINT2YR, replace=True )
            ageMeanList += [test_sample.age.mean()]
        self.mean = np.mean(ageMeanList)
        self.sd = np.std(ageMeanList)

    def test_people_from_population(self):
        test_pop = PopulationFactory.get_nhanes_population(n=self.test_n, year=2015, personFilters=None, nhanesWeights=True, distributions=False)
        test_people = test_pop._people

        test_ages = [x._age[0] for x in test_people]
        self.assertLess(np.mean(test_ages), self.mean + 2*self.sd)
        self.assertGreater(np.mean(test_ages), self.mean - 2*self.sd)

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
