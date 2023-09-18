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
from microsim.trials.risk_filter import RiskFilter
from microsim.trials.trial_utils import randomizationSchema
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.outcome_model_type import OutcomeModelType
import random


class TestBasicTrialOperations(unittest.TestCase):
    def setUp(self):  
        self.popSize = 100
        self.ageThreshold = 40
        self.targetPopulation = NHANESDirectSamplePopulation(self.popSize, 1999, rng = np.random.default_rng())
        self.trialDescription = TrialDescription(sampleSizes=[self.popSize], 
                                                durations=[10], 
                                                inclusionFilters=[lambda x : x._age[0] > self.ageThreshold], 
                                                exclusionFilters=None, 
                                                analyses=None,
                                                #randomizationSchema=lambda x : np.random.uniform() < 0.5,
                                                randomizationSchema=randomizationSchema,
                                                treatment=None)

        self.oldJoe = Person(
            age=60,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=140,
            dbp=90,
            a1c=5.5,
            hdl=50,
            totChol=200,
            bmi=25,
            ldl=90,
            trig=150,
            waist=45,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0,
            initializeAfib=lambda x: False,
            rng = np.random.default_rng())    
        # advance him one year to get an additional GCP value
        self.oldJoe.advance_year(CohortRiskModelRepository(), OutcomeModelRepository(), rng = np.random.default_rng())
        self.oldJoe._antiHypertensiveCount[-1] = 0


    def test_risk_filter(self):
        # true cv risk for patient is 0.005321896646249357
        # true dementia risk for patient is 0.00013823232976419768

        self.assertTrue(RiskFilter({OutcomeModelType.DEMENTIA: 0.00001}).exceedsThresholds(self.oldJoe))
        self.assertFalse(RiskFilter({OutcomeModelType.DEMENTIA: 0.01}).exceedsThresholds(self.oldJoe))
        
        self.assertTrue(RiskFilter({OutcomeModelType.CARDIOVASCULAR: 0.004}).exceedsThresholds(self.oldJoe))
        self.assertFalse(RiskFilter({OutcomeModelType.CARDIOVASCULAR: 0.006}).exceedsThresholds(self.oldJoe))

    
    # design a simple trial to only include patients over 40
    def test_simple_trial_inclusion(self):
        testTrial = Trial(self.trialDescription, self.targetPopulation, rng = np.random.default_rng())
        #self.assertEqual(self.popSize, len(testTrial.trialPopulation._people))
        # given that the populations involve ranodmization, we can't come up with a definite value
        # so, the approximate 99% CI on 50/100 has a LCI of 35...
        #self.assertGreaterEqual(65, len(testTrial.treatedPop._people))
        #self.assertLessEqual(35, len(testTrial.treatedPop._people))
        #self.assertGreaterEqual(65, len(testTrial.untreatePop._people))
        #self.assertLessEqual(35, len(testTrial.untreatePop._people))

        for i, person in testTrial.trialPopulation._people.items():
            self.assertGreaterEqual(person._age[0],  self.ageThreshold)

if __name__ == "__main__":
    unittest.main()
