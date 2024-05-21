import unittest
import numpy as np
import pandas as pd

from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.dementia_model import DementiaModel
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial import Trial
from microsim.trials.trial_type import TrialType
from microsim.outcome_model_repository import OutcomeModelRepository
import random
from microsim.treatment import DefaultTreatmentsType
from microsim.population_factory import PopulationFactory
from microsim.person_factory import PersonFactory
from microsim.dementia_model_repository import DementiaModelRepository
from microsim.cv_model_repository import CVModelRepository
from microsim.person_filter import PersonFilter
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.population_model_repository import PopulationRepositoryType

class TestBasicTrialOperations(unittest.TestCase):
    def setUp(self):  
        self.popSize = 100
        self.ageThreshold = 40
        self.agePf = PersonFilter(addCommonFilters=False)
        self.agePf.add_filter("df", "lowAgeLimit", lambda x: x[DynamicRiskFactorsType.AGE.value]>self.ageThreshold)
        self.riskPf = PersonFilter(addCommonFilters=False)
        self.riskPf.add_filter("person",
                               "dementiaLowLimit", 
                               lambda x: DementiaModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x, years=1)>0.00001)
        self.riskPf.add_filter("person",
                               "dementiaHighLimit", 
                               lambda x: DementiaModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x, years=1)<0.01)
        self.riskPf.add_filter("person",
                               "cvHighLimit",
                               lambda x: CVModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x)<0.006)
        self.riskPf.add_filter("person",
                               "cvLowLimit",
                               lambda x: CVModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x)>0.004)
 
        self.trialDescription = NhanesTrialDescription(trialType=TrialType.COMPLETELY_RANDOMIZED,
                                                       blockFactors=list(),
                                                       sampleSize = self.popSize,
                                                       duration=10,
                                                       treatmentStrategies = "noTreatment",
                                                       nWorkers=1,
                                                       personFilters=self.agePf,
                                                       year=1999, nhanesWeights=True, distributions=False)

        self.x = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 60,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 140,
                               DynamicRiskFactorsType.DBP.value: 90,
                               DynamicRiskFactorsType.A1C.value: 5.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 200,
                               DynamicRiskFactorsType.BMI.value: 25,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 45,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "oldJoe"}, index=[0])

        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()
        self.oldJoe = PersonFactory.get_nhanes_person(self.x.iloc[0], initializationModelRepository)
        self.oldJoe._afib = [False]
        # advance him one year to get an additional GCP value
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()._repository
        self.oldJoe.advance(1, popModelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
                               popModelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
                               popModelRepository[PopulationRepositoryType.OUTCOMES.value],
                               None)
        self.oldJoe._antiHypertensiveCount[-1] = 0

    def test_risk_filter(self):
        # true cv risk for patient is 0.005321896646249357
        # true dementia risk for patient is 0.00013823232976419768
        for filterFunction in self.riskPf.filters["person"].values():    
            self.assertTrue(map(filterFunction, [self.oldJoe]))

    # design a simple trial to only include patients over 40
    def test_simple_trial_inclusion(self):
        #testTrial = Trial(self.trialDescription, self.targetPopulation, rng = np.random.default_rng())
        tr = Trial(self.trialDescription) 
      
        #self.assertEqual(self.popSize, len(testTrial.trialPopulation._people))
        # given that the populations involve ranodmization, we can't come up with a definite value
        # so, the approximate 99% CI on 50/100 has a LCI of 35...
        #self.assertGreaterEqual(65, len(testTrial.treatedPop._people))
        #self.assertLessEqual(35, len(testTrial.treatedPop._people))
        #self.assertGreaterEqual(65, len(testTrial.untreatePop._people))
        #self.assertLessEqual(35, len(testTrial.untreatePop._people))

        for person in tr.treatedPop._people:
            self.assertGreaterEqual(person._age[0],  self.ageThreshold)
        for person in tr.controlPop._people:
            self.assertGreaterEqual(person._age[0],  self.ageThreshold)

if __name__ == "__main__":
    unittest.main()
