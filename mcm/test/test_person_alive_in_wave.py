from mcm.person import Person
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.outcome_model_repository import OutcomeModelRepository
from mcm.outcome import Outcome, Outcome
from mcm.outcome import OutcomeType
from mcm.education import Education
from mcm.test.test_risk_model_repository import TestRiskModelRepository

from mcm.smoking_status import SmokingStatus
import unittest


def initializeAFib(person):
    return None


class AgeOver50CausesFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()

    # override super to alays return a probability of each outcom eas 1
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, 1) if person._age[-1] > 50 else None

    def assign_non_cv_mortality(self, person, years=1):
        return False


class AgeOver50CausesNonCVMortality(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()

    # override super to alays return a probability of each outcom eas 1
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return None

    def assign_non_cv_mortality(self, person, years=1):
        return True if person._age[-1] > 50 else False


class TestPersonAdvanceOutcomes(unittest.TestCase):

    def setUp(self):
        self.oldJoe = Person(age=60, gender=NHANESGender.MALE,
                             raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
                             sbp=140, dbp=90, a1c=5.5, hdl=50, totChol=200, bmi=25, ldl=90,
                             trig=150, waist=45, anyPhysicalActivity=0,
                             education=Education.COLLEGEGRADUATE,
                             smokingStatus=SmokingStatus.NEVER,
                             antiHypertensiveCount=0, statin=0, otherLipidLoweringMedicationCount=0,
                             initializeAfib=initializeAFib)

        self.youngJoe = Person(age=40, gender=NHANESGender.MALE,
                               raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
                               sbp=140, dbp=90, a1c=5.5, hdl=50, totChol=200, bmi=25, ldl=90,
                               trig=150, waist=45, anyPhysicalActivity=0,
                               education=Education.COLLEGEGRADUATE,
                               smokingStatus=SmokingStatus.NEVER,
                               antiHypertensiveCount=0, statin=0, 
                               otherLipidLoweringMedicationCount=0, initializeAfib=initializeAFib)
    
    def testFatalStrokeLeadsToConsistentEffects(self):
        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesFatalStroke())
        self.assertEqual(41, self.youngJoe._age[-1])
        self.assertEqual(False, self.youngJoe.is_dead())
        self.assertEqual(False, self.youngJoe._stroke)

        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(0))
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(1))
        with self.assertRaises(Exception):
            self.youngJoe.alive_at_start_of_wave(2)

        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesFatalStroke())
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(2))

        self.oldJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesFatalStroke())
        self.assertEqual(60, self.oldJoe._age[-1])
        self.assertEqual(True, self.oldJoe.is_dead())
        self.assertEqual(True, self.oldJoe._stroke)

        self.assertEqual(True, self.oldJoe.alive_at_start_of_wave(0))
        self.assertEqual(False, self.oldJoe.alive_at_start_of_wave(1))
        
        # this is called to verify that it DOES NOT throw an excepiotn
        self.oldJoe.alive_at_start_of_wave(2)

    def testNoCVMortalityLeadsToConsistentEffects(self):
        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesNonCVMortality())
        self.assertEqual(41, self.youngJoe._age[-1])
        self.assertEqual(False, self.youngJoe.is_dead())
        self.assertEqual(False, self.youngJoe._stroke)

        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(0))
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(1))
        with self.assertRaises(Exception):
            self.youngJoe.alive_at_start_of_wave(2)

        self.youngJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesNonCVMortality())
        self.assertEqual(True, self.youngJoe.alive_at_start_of_wave(2))

        self.oldJoe.advance_year(TestRiskModelRepository(), AgeOver50CausesNonCVMortality())
        self.assertEqual(60, self.oldJoe._age[-1])
        self.assertEqual(True, self.oldJoe.is_dead())
        self.assertEqual(False, self.oldJoe._stroke)

        self.assertEqual(True, self.oldJoe.alive_at_start_of_wave(0))
        self.assertEqual(False, self.oldJoe.alive_at_start_of_wave(1))
        
        # this is called to verify that it DOES NOT throw an excepiotn
        self.oldJoe.alive_at_start_of_wave(2)

if __name__ == "__main__":
    unittest.main()
