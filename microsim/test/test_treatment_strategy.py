import unittest
from microsim.initialization_repository import InitializationRepository
import numpy as np
import pandas as pd

from microsim.person import Person
from microsim.education import Education
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.bp_treatment_strategies import (
    AddBPTreatmentMedsToGoal120,
    AddASingleBPMedTreatmentStrategy,
    jnc8ForHighRiskLowBpTarget
)
from microsim.population_factory import PopulationFactory
from microsim.population import Population
from microsim.person_factory import PersonFactory
from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import RaceEthnicity
from microsim.treatment import DefaultTreatmentsType
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.test.outcome_models_repositories import NoOutcome
from microsim.treatment_strategy_repository import TreatmentStrategyRepository
from microsim.population_model_repository import PopulationRepositoryType
from microsim.treatment import TreatmentStrategyStatus

class TestTreatmentStrategy(unittest.TestCase):

    def getPerson(self, baselineSBP=140, baselineDBP=80):
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()
        x = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 75,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.MEXICAN_AMERICAN.value,
                               DynamicRiskFactorsType.SBP.value: baselineSBP,
                               DynamicRiskFactorsType.DBP.value: baselineDBP,
                               DynamicRiskFactorsType.A1C.value: 6.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 210,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 50,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.FORMER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "person"}, index=[0])
        person = PersonFactory.get_nhanes_person(x.iloc[0], initializationModelRepository)
        person._afib = [False]
        return person

    def setUp(self):
        self.baselineSBP = 140
        self.baselineDBP = 80
        self.singleMedStrategy = TreatmentStrategyRepository()
        self.singleMedStrategy._repository["bp"] = AddASingleBPMedTreatmentStrategy()
        self._test_person = self.getPerson()
        self._risk_model_repository = TestRiskModelRepository(nullModels=True)
        # setup so that the SBP  always stays the same
        self._risk_model_repository.set_default_model_for_name("sbp")
        self._risk_model_repository._repository["sbp"]._params["age"] = 0
        self._risk_model_repository._repository["sbp"]._params["sbp"] = 1.0
        self._risk_model_repository._repository["sbp"]._params["intercept"] = 0
        self._risk_model_repository.set_default_model_for_name("dbp")
        self._risk_model_repository._repository["dbp"]._params["age"] = 0
        self._risk_model_repository._repository["dbp"]._params["dbp"] = 1.0
        self._risk_model_repository._repository["dbp"]._params["sbp"] = 0
        self._risk_model_repository._repository["dbp"]._params["intercept"] = 0

        self._outcome_model_repository = NoOutcome()

    def add_a_single_blood_pressure_medication_strategy(person):
        return {"_antiHypertensiveCount": 1}, {"_sbp": -5, "_dbp": -3}, {}

    def testSimpleBPTreatmentStrategy(self):
        self._test_person.advance(2, self._risk_model_repository, self._risk_model_repository, self._outcome_model_repository, None)

        self.assertEqual(self.baselineSBP, self._test_person._sbp[1])
        self.assertEqual(self.baselineDBP, self._test_person._dbp[1])
        self.assertEqual(0, self._test_person._antiHypertensiveCount[1])

        self._test_person.advance(1, self._risk_model_repository, self._risk_model_repository, self._outcome_model_repository, self.singleMedStrategy)

        self.assertEqual(
            self.baselineSBP - self.singleMedStrategy._repository["bp"].sbpLowering, self._test_person._sbp[2]
        )
        self.assertEqual(
            self.baselineDBP - self.singleMedStrategy._repository["bp"].dbpLowering, self._test_person._dbp[2]
        )
        self.assertEqual(0, self._test_person._antiHypertensiveCount[2])
        self.assertEqual(1, self._test_person._treatmentStrategies["bp"]["bpMedsAdded"])

    def testTreatTo12080Strategy(self):
        dbpAtGoal = self.getPerson(190, 65)
        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = AddBPTreatmentMedsToGoal120()
        dbpAtGoal.advance(1, self._risk_model_repository, self._risk_model_repository, self._outcome_model_repository, tsr)

        # no BP meds because baseline DBP is 80
        self.assertEqual(0, self._test_person._antiHypertensiveCount[-1])

        highSbp = self.getPerson(190, 150)
        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = AddBPTreatmentMedsToGoal120()
        highSbp.advance(1, self._risk_model_repository, self._risk_model_repository, self._outcome_model_repository, tsr)
        # 190-120 / 5.5
        self.assertEqual(12, highSbp._treatmentStrategies["bp"]["bpMedsAdded"])
        self.assertEqual(124, highSbp._sbp[-1])

        mediumSBP = self.getPerson(140, 150)
        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = AddBPTreatmentMedsToGoal120()
        mediumSBP.advance(1, self._risk_model_repository, self._risk_model_repository, self._outcome_model_repository, tsr)   
        # 140-120 / 5.5
        self.assertEqual(3, mediumSBP._treatmentStrategies["bp"]["bpMedsAdded"])
        self.assertEqual(123.5, mediumSBP._sbp[-1])

        lowSBP = self.getPerson(110, 100)
        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = AddBPTreatmentMedsToGoal120()
        lowSBP.advance(1, self._risk_model_repository, self._risk_model_repository, self._outcome_model_repository, tsr)
        self.assertEqual(0, lowSBP._antiHypertensiveCount[-1])
        self.assertEqual(110, lowSBP._sbp[-1])

        dbpDrives = self.getPerson(250, 110)
        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = AddBPTreatmentMedsToGoal120()
        dbpDrives.advance(1, self._risk_model_repository, self._risk_model_repository, self._outcome_model_repository, tsr)
        # 110-66/3.1
        self.assertEqual(14, dbpDrives._treatmentStrategies["bp"]["bpMedsAdded"])
        self.assertEqual(66.6, dbpDrives._dbp[-1])

    def testSprintStrategyAtPopLevelOneWave(self):
        maxMedPerson = self.getPerson(190, 120)
        people = PopulationFactory.get_cloned_people(maxMedPerson, 100)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        pop = Population(people, popModelRepository)
        pop._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value] = TestRiskModelRepository(nullModels=True)
        pop._modelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value] = TestRiskModelRepository(nullModels=True)
        pop._modelRepository[PopulationRepositoryType.OUTCOMES.value] = NoOutcome()

        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = jnc8ForHighRiskLowBpTarget(0.075, {'sbp' : 126, 'dbp': 0})

        pop.advance(1, treatmentStrategies = tsr)
        tsr._repository["bp"].status = TreatmentStrategyStatus.MAINTAIN
        pop.advance(1, treatmentStrategies = tsr)

        for j in range(0,100): #our population consists of 100 persons, check all of them
            # checking that anti hypertensives weren't accidentally added
            self.assertEqual(0, pop._people.iloc[j]._antiHypertensiveCount[-1])
            # should add 4 bp meds for everybody
            self.assertEqual(4, pop._people.iloc[j]._treatmentStrategies["bp"]["bpMedsAdded"])
            # BP should be lowered by 4*BP lowering effect
            self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, pop._people.iloc[j]._sbp[-1])


    def testSprintStrategyAtPopLevelTwoWave(self):
        maxMedPerson = self.getPerson(190, 120)
        people = PopulationFactory.get_cloned_people(maxMedPerson, 100)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        pop = Population(people, popModelRepository)
        pop._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value] = TestRiskModelRepository(nullModels=True)
        pop._modelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value] = TestRiskModelRepository(nullModels=True)
        pop._modelRepository[PopulationRepositoryType.OUTCOMES.value] = NoOutcome()

        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = jnc8ForHighRiskLowBpTarget(0.075, {'sbp' : 126, 'dbp': 0})

        pop.advance(1, treatmentStrategies = tsr)
        tsr._repository["bp"].status = TreatmentStrategyStatus.MAINTAIN
        pop.advance(2, treatmentStrategies = tsr)

        for j in range(0,100): #our population consists of 100 persons, check all of them
            self.assertEqual(0, pop._people.iloc[j]._antiHypertensiveCount[-1])
            self.assertEqual(4, pop._people.iloc[j]._treatmentStrategies["bp"]["bpMedsAdded"])
            #self.assertEqual(4, np.array(pop._people.iloc[j]._bpMedsAdded).sum())
            self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, pop._people.iloc[j]._sbp[-1])

    def testSprintStrategyAtPopLevelFiveWave(self):
        maxMedPerson = self.getPerson(190, 120)
        people = PopulationFactory.get_cloned_people(maxMedPerson, 100)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        pop = Population(people, popModelRepository)
        pop._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value] = TestRiskModelRepository(nullModels=True)
        pop._modelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value] = TestRiskModelRepository(nullModels=True)
        pop._modelRepository[PopulationRepositoryType.OUTCOMES.value] = NoOutcome()

        tsr = TreatmentStrategyRepository()
        tsr._repository["bp"] = jnc8ForHighRiskLowBpTarget(0.075, {'sbp' : 126, 'dbp': 0})

        pop.advance(1, treatmentStrategies = tsr)
        tsr._repository["bp"].status = TreatmentStrategyStatus.MAINTAIN
        pop.advance(5, treatmentStrategies = tsr)

        for j in range(0,100): #our population consists of 100 persons, check all of them
            self.assertEqual(4, pop._people.iloc[j]._treatmentStrategies["bp"]["bpMedsAdded"])
            for wave in range(2, 5):
                self.assertEqual(0, pop._people.iloc[j]._antiHypertensiveCount[wave])
                #self.assertEqual(4, np.array(pop._people.iloc[j]._bpMedsAdded).sum())
                self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, pop._people.iloc[j]._sbp[wave])

if __name__ == "__main__":
    unittest.main()
