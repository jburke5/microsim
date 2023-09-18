import unittest
from microsim.initialization_repository import InitializationRepository
import numpy as np

from microsim.person import Person
from microsim.education import Education
from microsim.population import ClonePopulation
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.bp_treatment_strategies import (
    AddBPTreatmentMedsToGoal120,
    AddASingleBPMedTreatmentStrategy,
    jnc8ForHighRiskLowBpTarget
)


class TestTreatmentStrategy(unittest.TestCase):
    def initializeAfib(person):
        return 0 #modified to 0 from None, because afib was utilized as part of a model

    def getPerson(self, baselineSBP=140, baselineDBP=80):
        return Person(
            age=75,
            gender=1,
            raceEthnicity=1,
            sbp=baselineSBP,
            dbp=baselineDBP,
            a1c=6.5,
            hdl=50,
            totChol=210,
            ldl=90,
            trig=150,
            bmi=22,
            waist=50,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=1,
            alcohol=0,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=TestTreatmentStrategy.initializeAfib,
            initializationRepository=InitializationRepository(),
            rng = np.random.default_rng(),
            randomEffects={'gcp' : 0,
                           'gcpStroke': 0,
                           'gcpStrokeSlope': 0}
        )

    def setUp(self):
        self.baselineSBP = 140
        self.baselineDBP = 80
        self.singleMedStrategy = AddASingleBPMedTreatmentStrategy()
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

    def add_a_single_blood_pressure_medication_strategy(person):
        return {"_antiHypertensiveCount": 1}, {"_sbp": -5, "_dbp": -3}, {}

    def testSimpleBPTreatmentStrategy(self):
        self._test_person.advance_treatment(self._risk_model_repository)
        self._test_person.advance_risk_factors(self._risk_model_repository, rng = np.random.default_rng())

        self.assertEqual(self.baselineSBP, self._test_person._sbp[1])
        self.assertEqual(self.baselineDBP, self._test_person._dbp[1])
        self.assertEqual(0, self._test_person._antiHypertensiveCount[1])

        self._test_person._bpTreatmentStrategy = self.singleMedStrategy

        self._test_person.advance_treatment(self._risk_model_repository)
        self._test_person.advance_risk_factors(self._risk_model_repository, rng = np.random.default_rng())

        self.assertEqual(
            self.baselineSBP - self.singleMedStrategy.sbpLowering, self._test_person._sbp[2]
        )
        self.assertEqual(
            self.baselineDBP - self.singleMedStrategy.dbpLowering, self._test_person._dbp[2]
        )
        self.assertEqual(1, self._test_person._antiHypertensiveCount[2])

    def testTreatTo12080Strategy(self):
        dbpAtGoal = self.getPerson(190, 65)
        dbpAtGoal._bpTreatmentStrategy = AddBPTreatmentMedsToGoal120()
        dbpAtGoal.advance_treatment(self._risk_model_repository)
        # no BP meds because baseline DBP is 80
        self.assertEqual(0, self._test_person._antiHypertensiveCount[-1])

        highSbp = self.getPerson(190, 150)
        highSbp._bpTreatmentStrategy = AddBPTreatmentMedsToGoal120()
        highSbp.advance_treatment(self._risk_model_repository)
        # 190-120 / 5.5
        self.assertEqual(12, highSbp._bpMedsAdded[-1])
        self.assertEqual(124, highSbp._sbp[-1])

        mediumSBP = self.getPerson(140, 150)
        mediumSBP._bpTreatmentStrategy = AddBPTreatmentMedsToGoal120()
        mediumSBP.advance_treatment(self._risk_model_repository)
        # 140-120 / 5.5
        self.assertEqual(3, mediumSBP._bpMedsAdded[-1])
        self.assertEqual(123.5, mediumSBP._sbp[-1])

        lowSBP = self.getPerson(110, 100)
        lowSBP._bpTreatmentStrategy = AddBPTreatmentMedsToGoal120()
        lowSBP.advance_treatment(self._risk_model_repository)
        self.assertEqual(0, lowSBP._antiHypertensiveCount[-1])
        self.assertEqual(110, lowSBP._sbp[-1])

        dbpDrives = self.getPerson(250, 110)
        dbpDrives._bpTreatmentStrategy = AddBPTreatmentMedsToGoal120()
        dbpDrives.advance_treatment(self._risk_model_repository)
        # 110-66/3.1
        self.assertEqual(14, dbpDrives._bpMedsAdded[-1])
        self.assertEqual(66.6, dbpDrives._dbp[-1])

    def testSprintStrategyAtPopLevelOneWave(self):
        maxMedPerson = self.getPerson(190, 120)
        pop = ClonePopulation(maxMedPerson, 100)
        pop._risk_model_repository = TestRiskModelRepository(nullModels=True)

        pop.set_bp_treatment_strategy(jnc8ForHighRiskLowBpTarget(0.075, {'sbp' : 126, 'dbp': 0}))
        alive, df = pop.advance_vectorized(1, rng = np.random.default_rng())

        # checking that anti hypertensives weren't accidentally added
        self.assertEqual(0, alive.antiHypertensiveCount.max())
        # should add 4 bp meds for everybody
        self.assertEqual(4, alive.bpMedsAdded.max())
        self.assertEqual(4, alive.bpMedsAdded.min())
        # should add 4 bp meds for everybody
        self.assertEqual(4, alive.totalBPMedsAdded.max())
        self.assertEqual(4, alive.totalBPMedsAdded.min())
        # BP should be lowered by 4*BP lowering effect
        self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, alive.sbp.max())
        self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, alive.sbp.min())

        for j in range(0,100): #our population consists of 100 persons, check all of them
            if pop._people.iloc[j]._age[-1]==75+1: #if person is alive, then check
                self.assertEqual(0, pop._people.iloc[j]._antiHypertensiveCount[-1])
                self.assertEqual(4, pop._people.iloc[j]._bpMedsAdded[-1])
                self.assertEqual(4, np.array(pop._people.iloc[j]._bpMedsAdded).sum())
                self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, pop._people.iloc[j]._sbp[-1])


    def testSprintStrategyAtPopLevelTwoWave(self):
        maxMedPerson = self.getPerson(190, 120)
        pop = ClonePopulation(maxMedPerson, 100)
        pop._risk_model_repository = TestRiskModelRepository(nullModels=True)
        pop.set_bp_treatment_strategy(jnc8ForHighRiskLowBpTarget(0.075, {'sbp' : 126, 'dbp': 0}))
        alive, df = pop.advance_vectorized(2, rng = np.random.default_rng())

        # checking that anti hypertensives weren't accidentally added
        self.assertEqual(0, alive.antiHypertensiveCount.max())
        # should add no new medicatinos
        self.assertEqual(0, alive.bpMedsAdded.max())
        self.assertEqual(0, alive.bpMedsAdded.min())
        # should add 4 bp meds for everybody
        self.assertEqual(4, alive.totalBPMedsAdded.max())
        self.assertEqual(4, alive.totalBPMedsAdded.min())
        # BP should be lowered by 4*BP lowering effect
        self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, alive.sbp.max())
        self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, alive.sbp.min())

        for j in range(0,100): #our population consists of 100 persons, check all of them
            if pop._people.iloc[j]._age[-1]==75+2: #if person is alive, then check
                self.assertEqual(0, pop._people.iloc[j]._antiHypertensiveCount[-1])
                self.assertEqual(0, pop._people.iloc[j]._bpMedsAdded[-1])
                self.assertEqual(4, np.array(pop._people.iloc[j]._bpMedsAdded).sum())
                self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, pop._people.iloc[j]._sbp[-1])


    def testSprintStrategyAtPopLevelFiveWave(self):
        maxMedPerson = self.getPerson(190, 120)
        pop = ClonePopulation(maxMedPerson, 100)
        pop._risk_model_repository = TestRiskModelRepository(nullModels=True)
        pop.set_bp_treatment_strategy(jnc8ForHighRiskLowBpTarget(0.075, {'sbp' : 126, 'dbp': 0}))
        alive, df = pop.advance_vectorized(5, rng = np.random.default_rng())

        for j in range(0,100): #our population consists of 100 persons, check all of them
            if pop._people.iloc[j]._age[-1]==75+5: #if person is alive, then check
                for wave in range(2, 5):
                     self.assertEqual(0, pop._people.iloc[j]._antiHypertensiveCount[wave])
                     self.assertEqual(0, pop._people.iloc[j]._bpMedsAdded[wave])
                     self.assertEqual(4, np.array(pop._people.iloc[j]._bpMedsAdded).sum())
                     self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, pop._people.iloc[j]._sbp[wave])

        # checking that anti hypertensives weren't accidentally added
        self.assertEqual(0, alive.antiHypertensiveCount.max())
        # should add no new medicatinos
        self.assertEqual(0, alive.bpMedsAdded.max())
        self.assertEqual(0, alive.bpMedsAdded.min())
        # should add 4 bp meds for everybody
        self.assertEqual(4, alive.totalBPMedsAdded.max())
        self.assertEqual(4, alive.totalBPMedsAdded.min())
        # BP should be lowered by 4*BP lowering effect
        self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, alive.sbp.max())
        self.assertEqual(190 - 4 * AddASingleBPMedTreatmentStrategy.sbpLowering, alive.sbp.min())

    

if __name__ == "__main__":
    unittest.main()
