import unittest
import pandas as pd
import numpy as np
from microsim.population import NHANESAgeStandardPopulation, NHANESDirectSamplePopulation

from microsim.person import Person
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome import Outcome
from microsim.outcome import OutcomeType


class AlwaysNonFatalMIOutcomeRepository(OutcomeModelRepository):
    def assign_cv_outcome(self, person):
        return Outcome(OutcomeType.MI, False)

    def assign_cv_outcome_vectorized(self, person):
        person.miNext = True
        person.strokeNext = False
        person.deadNext = False
        person.ageAtFirstMI = (
            person.age
            if (person.ageAtFirstMI is None) or (np.isnan(person.ageAtFirstMI))
            else person.ageAtFirstMI
        )
        return person

    def assign_non_cv_mortality(self, person):
        return False

    def assign_non_cv_mortality_vectorized(self, person, years=1):
        return False


class TestPopulationReporting(unittest.TestCase):
    def setUp(self):
        self.popSize = 10000
        self.pop1 = NHANESAgeStandardPopulation(self.popSize, 1999)

    def get_event_count(self, pop, outcomeType, wave):
        return pd.Series(
            [
                person.has_outcome_during_wave(wave, outcomeType)
                for i, person in pop._people.iteritems()
            ]
        ).sum()

    def testAllCasesHaveEvent(self):
        self.pop1._outcome_model_repository = AlwaysNonFatalMIOutcomeRepository()

        self.pop1.advance_vectorized(1)
        self.assertEqual(self.popSize, self.get_event_count(self.pop1, OutcomeType.MI, 1))

        # events per 100000 = 100000
        self.assertAlmostEqual(
            100000,
            self.pop1.calculate_mean_age_sex_standardized_incidence(OutcomeType.MI)[0],
            delta=0.01,
        )
        self.assertAlmostEqual(
            0,
            self.pop1.calculate_mean_age_sex_standardized_incidence(OutcomeType.STROKE)[0],
            delta=0.01,
        )
        self.assertAlmostEqual(
            0, self.pop1.calculate_mean_age_sex_standardized_mortality(), delta=0.01
        )

        self.pop1.reset_to_baseline()
        self.pop1.advance_vectorized(1)
        self.assertAlmostEqual(
            100000,
            self.pop1.calculate_mean_age_sex_standardized_incidence(OutcomeType.MI)[0],
            delta=0.01,
        )
        self.assertAlmostEqual(
            0,
            self.pop1.calculate_mean_age_sex_standardized_incidence(OutcomeType.STROKE)[0],
            delta=0.01,
        )
        self.assertAlmostEqual(
            0, self.pop1.calculate_mean_age_sex_standardized_mortality(), delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
