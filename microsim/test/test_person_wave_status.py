from microsim.person import Person
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome import Outcome
from microsim.alcohol_category import AlcoholCategory
from microsim.outcome import OutcomeType
from microsim.education import Education
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.gcp_model import GCPModel
from microsim.dementia_model import DementiaModel
from microsim.outcome_model_type import OutcomeModelType

from microsim.smoking_status import SmokingStatus
import unittest


def initializeAFib(person):
    return None


class AgeOver50CausesFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()
        self._models = {}
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()
        self._models[OutcomeModelType.DEMENTIA] = DementiaModel()

    # override super to alays return a probability of each outcom eas 1
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, 1) if person._age[-1] > 50 else None

    def assign_non_cv_mortality(self, person, years=1):
        return False

    def get_gcp(self, person):
        return 50


class NonFatalStrokeAndNonCVMortality(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()
        self._models = {}
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()
        self._models[OutcomeModelType.DEMENTIA] = DementiaModel()

    # override super to alays return a probability of each outcom eas 1
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, 0)

    def assign_non_cv_mortality(self, person, years=1):
        return True

    def get_gcp(self, person):
        return 50


class AgeOver50CausesNonCVMortality(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()
        self._models = {}
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()
        self._models[OutcomeModelType.DEMENTIA] = DementiaModel()

    # override super to alays return a probability of each outcom eas 1
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return None

    def assign_non_cv_mortality(self, person, years=1):
        return True if person._age[-1] > 50 else False

    def get_gcp(self, person):
        return 50


class TestPersonWaveStatus(unittest.TestCase):
    def setUp(self):
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
            creatinine = 0,
            initializeAfib=initializeAFib,
        )

        self.youngJoe = Person(
            age=40,
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
            creatinine = 0,
            initializeAfib=initializeAFib,
        )

    def testStatusAfterFatalStroke(self):
        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesFatalStroke())
        self.assertEqual(41, self.youngJoe._age[-1])
        self.assertEqual(False, self.youngJoe.is_dead())
        self.assertEqual(False, self.youngJoe._stroke)

        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(1))
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(2))
        with self.assertRaises(Exception):
            self.youngJoe.alive_at_start_of_wave(3)

        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesFatalStroke())
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(3))

        self.oldJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesFatalStroke())
        self.assertEqual(60, self.oldJoe._age[-1])
        self.assertEqual(True, self.oldJoe.is_dead())
        self.assertEqual(True, self.oldJoe._stroke)

        self.assertEqual(True, self.oldJoe.alive_at_start_of_wave(1))
        self.assertEqual(False, self.oldJoe.alive_at_start_of_wave(2))

        # this is called to verify that it DOES NOT throw an excepiotn
        self.oldJoe.alive_at_start_of_wave(3)

    def testNonCVMortalityLeadsToCorrectStatus(self):
        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesNonCVMortality())
        self.assertEqual(41, self.youngJoe._age[-1])
        self.assertEqual(False, self.youngJoe.is_dead())
        self.assertEqual(False, self.youngJoe._stroke)

        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(1))
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(2))
        with self.assertRaises(Exception):
            self.youngJoe.alive_at_start_of_wave(3)

        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesNonCVMortality())
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(3))

        self.oldJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesNonCVMortality())
        self.assertEqual(60, self.oldJoe._age[-1])
        self.assertEqual(True, self.oldJoe.is_dead())
        self.assertEqual(False, self.oldJoe._stroke)

        self.assertEqual(True, self.oldJoe.alive_at_start_of_wave(1))
        self.assertEqual(False, self.oldJoe.alive_at_start_of_wave(2))

        # this is called to verify that it DOES NOT throw an excepiotn
        self.oldJoe.alive_at_start_of_wave(3)

    def testHasFatalStrokeInWaveIsCaptured(self):
        self.oldJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesFatalStroke())
        self.assertEqual(True, self.oldJoe.has_stroke_during_simulation())
        self.assertEqual(True, self.oldJoe.has_stroke_during_wave(1))
        with self.assertRaises(Exception):
            self.oldJoe.has_stroke_during_wave(0)
        # calling to make sure it does NOT raise an exception...you should
        # be able to ask whether somebody has a stroke in a wave after they are dead, the answer is
        # "No"
        self.assertEqual(False, self.oldJoe.has_stroke_during_wave(2))

    def testNonFatalStrokeInWaveWithNonCVDeathIsCaptured(self):
        self.oldJoe.advance_year(TestRiskModelRepository(), NonFatalStrokeAndNonCVMortality())
        self.assertEqual(True, self.oldJoe.has_stroke_during_simulation())
        self.assertEqual(True, self.oldJoe.has_stroke_during_wave(1))
        self.assertEqual(True, self.oldJoe.is_dead())


if __name__ == "__main__":
    unittest.main()
