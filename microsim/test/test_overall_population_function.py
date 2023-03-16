import unittest
import pdb
from microsim.population import NHANESDirectSamplePopulation


class TestOverallPopulationFunction(unittest.TestCase):
    def test_basic_population(self):
        popSize = 1
        numYears = 3
        pop = NHANESDirectSamplePopulation(popSize, 1999)
        for i in range(1, numYears):
            pop.advance_vectorized(years=1)
        self.assertEqual(popSize, len(pop._people))
        if pop._people.iloc[0]._age[-1] == pop._people.iloc[0]._age[0] + numYears-1: #if person is alive, then check
           self.assertEqual(numYears, len(pop._people.iloc[0]._age))
