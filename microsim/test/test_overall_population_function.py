
import unittest

from microsim.population import NHANESDirectSamplePopulation


class TestOverallPopulationFunction(unittest.TestCase):
    def test_basic_population(self):
        popSize = 100
        numYears = 3
        pop = NHANESDirectSamplePopulation(popSize, 1999)
        for i in range(1, numYears):
            pop.advance_vectorized(years=1)

        self.assertEqual(popSize, len(pop._people))
        self.assertEqual(numYears, len(pop._people.iloc[0]._age))
