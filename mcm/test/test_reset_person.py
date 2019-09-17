import unittest

from mcm.person import Person
from mcm.test.test_risk_model_repository import TestRiskModelRepository
from mcm.outcome_model_repository import OutcomeModelRepository
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.education import Education
from mcm.smoking_status import SmokingStatus
from mcm.outcome import Outcome, OutcomeType


def initializeAfib(arg):
    return False


class AlwaysStrokeeOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysStrokeeOutcomeRepository, self).__init__()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, True)


class TestResetPerson(unittest.TestCase):
    def setUp(self):
        self.baseAge = 55
        self.baseSBP = 120
        self._white_male = Person(
            age=self.baseAge, gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, waist=34, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER, antiHypertensiveCount=0,
            statin=0, otherLipidLoweringMedicationCount=0, initializeAfib=initializeAfib)

        self._baseline_stroke_person = Person(
            age=self.baseAge, gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, waist=34, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER, antiHypertensiveCount=0,
            statin=0, otherLipidLoweringMedicationCount=0, initializeAfib=initializeAfib,
            selfReportStrokeAge=50)

        self._white_male.advance_year(TestRiskModelRepository(), AlwaysStrokeeOutcomeRepository())
        self._baseline_stroke_person.advance_year(TestRiskModelRepository(),
                                                  AlwaysStrokeeOutcomeRepository())

    def testResetBasicAttributes(self):
        self.assertEqual(2, len(self._white_male._dbp))

        self._white_male.reset_to_baseline()

        self.assertEqual(1, len(self._white_male._dbp))
        self.assertEqual(1, len(self._white_male._totChol))
        self.assertEqual(1, len(self._white_male._trig))
        self.assertEqual(1, len(self._white_male._trig))
        self.assertEqual(1, len(self._white_male._sbp))

        self.assertEqual(self.baseSBP, self._white_male._sbp[-1])
        self.assertEqual(self.baseAge, self._white_male._age[-1])

    def testResetOutcomes(self):
        self.assertEqual(1, len(self._white_male._outcomes[OutcomeType.STROKE]))
        self._white_male.reset_to_baseline()
        self.assertEqual(0, len(self._white_male._outcomes[OutcomeType.STROKE]))

    def testResetOutcomesPreservesPreSimOutcomes(self):
        self.assertEqual(2, len(self._baseline_stroke_person._outcomes[OutcomeType.STROKE]))
        self._baseline_stroke_person.reset_to_baseline()
        self.assertEqual(1, len(self._baseline_stroke_person._outcomes[OutcomeType.STROKE]))
