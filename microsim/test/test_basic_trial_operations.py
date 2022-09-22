import unittest
import numpy as np

from microsim.person import Person
from microsim.population import NHANESDirectSamplePopulation
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.dementia_model import DementiaModel
from microsim.trials.trial_description import TrialDescription
from microsim.trials.trial import Trial
import random


class TestBasicTrialOperations(unittest.TestCase):
    def setUp(self):  
        self.popSize = 100
        self.ageThreshold = 40
        self.targetPopulation = NHANESDirectSamplePopulation(self.popSize, 1999)
        self.trialDescription = TrialDescription(sampleSize=self.popSize, 
                                                duration=10, 
                                                inclusionFilter=lambda x : x._age[0] > self.ageThreshold, 
                                                exclusionFilter=None, 
                                                outcomes=None,
                                                randomizationSchema=lambda x : np.random.uniform() < 0.5)
        pass

    # design a simple trial to only include patients over 40
    def test_simple_trial_inclusion(self):
        testTrial = Trial(self.trialDescription, self.targetPopulation)
        #self.assertEqual(self.popSize, len(testTrial.trialPopulation._people))
        # given that the populations involve ranodmization, we can't come up with a definite value
        # so, the approximate 99% CI on 50/100 has a LCI of 35...
        #self.assertGreaterEqual(65, len(testTrial.treatedPop._people))
        #self.assertLessEqual(35, len(testTrial.treatedPop._people))
        #self.assertGreaterEqual(65, len(testTrial.untreatePop._people))
        #self.assertLessEqual(35, len(testTrial.untreatePop._people))

        for i, person in testTrial.trialPopulation._people.iteritems():
            self.assertGreaterEqual(person._age[0],  self.ageThreshold)

if __name__ == "__main__":
    unittest.main()