import unittest

from microsim.person import Person
from microsim.education import Education
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.bp_treatment_strategies import AddBPTreatmentMedsToGoal120, AddASingleBPMedTreatmentStrategy


class TestTreatmentStrategy(unittest.TestCase):

    def initializeAfib(person):
        return None

    def getPerson(self, baselineSBP=140, baselineDBP=80):
        return Person(
            age=75,
            gender=0,
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
            initializeAfib=TestTreatmentStrategy.initializeAfib)

    def setUp(self):
        self.baselineSBP = 140
        self.baselineDBP = 80
        self.singleMedStrategy = AddASingleBPMedTreatmentStrategy()
        self._test_person = self.getPerson()
        self._risk_model_repository = TestRiskModelRepository()
        # setup so that the SBP  always stays the same
        self._risk_model_repository._repository['sbp']._params['age'] = 0
        self._risk_model_repository._repository['sbp']._params['sbp'] = 1.0
        self._risk_model_repository._repository['sbp']._params['intercept'] = 0
        self._risk_model_repository._repository['dbp']._params['age'] = 0
        self._risk_model_repository._repository['dbp']._params['dbp'] = 1.0
        self._risk_model_repository._repository['dbp']._params['sbp'] = 0
        self._risk_model_repository._repository['dbp']._params['intercept'] = 0
        # setup so that the anti-hypertensive count stays at zero
        self._risk_model_repository._repository['antiHypertensiveCount']._params['age'] = 0
        self._risk_model_repository._repository['antiHypertensiveCount']._params['sbp'] = 0
        self._risk_model_repository._repository['antiHypertensiveCount']._params['intercept'] = 0

    def add_a_single_blood_pressure_medication_strategy(person):
        return {'_antiHypertensiveCount': 1}, {'_sbp': -5, '_dbp': -3}, {}

    def testSimpleBPTreatmentStrategy(self):
        self._test_person.advance_treatment(self._risk_model_repository)
        self._test_person.advance_risk_factors(self._risk_model_repository)

        self.assertEqual(self.baselineSBP, self._test_person._sbp[1])
        self.assertEqual(self.baselineDBP, self._test_person._dbp[1])
        self.assertEqual(0, self._test_person._antiHypertensiveCount[1])

        self._test_person._bpTreatmentStrategy = self.singleMedStrategy

        self._test_person.advance_treatment(self._risk_model_repository)
        self._test_person.advance_risk_factors(self._risk_model_repository)

        self.assertEqual(self.baselineSBP - self.singleMedStrategy.sbpLowering, self._test_person._sbp[2])
        self.assertEqual(self.baselineDBP - self.singleMedStrategy.dbpLowering, self._test_person._dbp[2])
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
        self.assertEqual(12, highSbp._antiHypertensiveCount[-1])
        self.assertEqual(124, highSbp._sbp[-1])

        mediumSBP = self.getPerson(140, 150)
        mediumSBP._bpTreatmentStrategy = AddBPTreatmentMedsToGoal120()
        mediumSBP.advance_treatment(self._risk_model_repository)
        # 140-120 / 5.5
        self.assertEqual(3, mediumSBP._antiHypertensiveCount[-1])
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
        self.assertEqual(14, dbpDrives._antiHypertensiveCount[-1])
        self.assertEqual(66.6, dbpDrives._dbp[-1])


if __name__ == "__main__":
    unittest.main()
