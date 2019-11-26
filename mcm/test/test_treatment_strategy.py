import unittest

from mcm.person import Person
from mcm.education import Education
from mcm.test.test_risk_model_repository import TestRiskModelRepository


class TestTreatmentStrategy(unittest.TestCase):

    def initializeAfib(person):
        return None

    def setUp(self):
        self.baselineSBP = 140
        self.baselineDBP = 80
        self._test_person = Person(
            age=75,
            gender=0,
            raceEthnicity=1,
            sbp=self.baselineSBP,
            dbp=self.baselineDBP,
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

        self._test_person._bpTreatmentStrategy = TestTreatmentStrategy.add_a_single_blood_pressure_medication_strategy

        self._test_person.advance_treatment(self._risk_model_repository)
        self._test_person.advance_risk_factors(self._risk_model_repository)

        self.assertEqual(self.baselineSBP - 5, self._test_person._sbp[2])
        self.assertEqual(self.baselineDBP - 3, self._test_person._dbp[2])
        self.assertEqual(1, self._test_person._antiHypertensiveCount[2])
