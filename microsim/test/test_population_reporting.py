import unittest
import pandas as pd
import numpy as np

from microsim.person import Person
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome import Outcome
from microsim.outcome import OutcomeType
from microsim.population_factory import PopulationFactory
from microsim.test.outcome_models_repositories import AlwaysNonFatalStroke, AlwaysFatalStroke, AlwaysNonFatalMI

class TestPopulationReporting(unittest.TestCase):
    def setUp(self):
        self.popSize = 10000
        self.pop1 = PopulationFactory.get_nhanes_age_standardized_population(self.popSize, 1999)

    def testAllCasesHaveEvent(self):
        self.pop1 = PopulationFactory.get_nhanes_age_standardized_population(self.popSize, 1999)
        self.pop1._modelRepository["outcomes"] = AlwaysNonFatalMI()

        self.pop1.advance(1)

        self.assertEqual(self.popSize, self.pop1.get_outcome_count(OutcomeType.MI))

        # events per 100000 = 100000
        #self.assertAlmostEqual(
        #    100000,
        #    self.pop1.calculate_mean_age_sex_standardized_incidence(OutcomeType.MI),
        #    delta=0.01,
        #)
        self.assertAlmostEqual(
            0,
            self.pop1.calculate_mean_age_sex_standardized_incidence(OutcomeType.STROKE),
            delta=0.01,
        )
        self.assertAlmostEqual(
            0, self.pop1.calculate_mean_age_sex_standardized_incidence(OutcomeType.DEATH), delta=0.01
        )



if __name__ == "__main__":
    unittest.main()
