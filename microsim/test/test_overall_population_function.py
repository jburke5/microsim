import unittest
import pdb
from microsim.population_factory import PopulationFactory
import numpy as np

class TestOverallPopulationFunction(unittest.TestCase):
    def test_basic_population(self):
        popSize = 10
        numYears = 3
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=1999, personFilters=None, nhanesWeights=True, distributions=False)
        for i in range(numYears):
            pop.advance(years=1)
        self.assertEqual(popSize, len(pop._people))
        for person in pop._people:
           if person.is_alive:
               self.assertEqual(numYears, len(person._age))

if __name__ == "__main__":
    unittest.main()
