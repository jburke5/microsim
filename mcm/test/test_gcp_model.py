import unittest

from mcm.person import Person
from mcm.education import Education
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity


class TestGCPModel(unittest.TestCase):

    def initializeAfib(person):
        return None

    def setUp(self):


        self._test_case_one = Person(
            age=75,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
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
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            initializeAfib=TestTreatmentStrategy.initializeAfib)


    def add_a_single_blood_pressure_medication_strategy(person):
 
