import unittest
import numpy as np

from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity


class TestCKDEquation(unittest.TestCase):
    def setUp(self):        
        self._black_female_high_cr = Person(
            age=52,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120,
            dbp=80,
            a1c=Person.convert_fasting_glucose_to_a1c(100),
            hdl=50,
            totChol=150,
            ldl=90,
            trig=150,
            bmi=26.6,
            waist=94,
            anyPhysicalActivity=1,
            education=Education.HIGHSCHOOLGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0.8,
            initializeAfib=TestCKDEquation.initializeAfib,
        )

        self._black_female_low_cr = Person(
            age=52,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120,
            dbp=80,
            a1c=Person.convert_fasting_glucose_to_a1c(100),
            hdl=50,
            totChol=150,
            ldl=90,
            trig=150,
            bmi=26.6,
            waist=94,
            anyPhysicalActivity=1,
            education=Education.HIGHSCHOOLGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0.4,
            initializeAfib=TestCKDEquation.initializeAfib,
        )

        self._white_male_high_cr = Person(
            age=52,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120,
            dbp=80,
            a1c=Person.convert_fasting_glucose_to_a1c(100),
            hdl=50,
            totChol=150,
            ldl=90,
            trig=150,
            bmi=26.6,
            waist=94,
            anyPhysicalActivity=1,
            education=Education.HIGHSCHOOLGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 1.2,
            initializeAfib=TestCKDEquation.initializeAfib,
        )
   
        self._white_male_low_cr = Person(
            age=52,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120,
            dbp=80,
            a1c=Person.convert_fasting_glucose_to_a1c(100),
            hdl=50,
            totChol=150,
            ldl=90,
            trig=150,
            bmi=26.6,
            waist=94,
            anyPhysicalActivity=1,
            education=Education.HIGHSCHOOLGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.ONETOSIX,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0.1,
            initializeAfib=TestCKDEquation.initializeAfib,
        )

    def initializeAfib(person):
        return None

    
    def testGFRs(self):
        blackFemaleHighCr = 166 * (self._black_female_high_cr._creatinine[0]/0.7)**-1.209 * 0.993**self._black_female_high_cr._age[-1]
        self.assertAlmostEqual(blackFemaleHighCr, self._black_female_high_cr._gfr)

        blackFemaleLowCr = 166 * (self._black_female_low_cr._creatinine[0]/0.7)**-.329 * 0.993**self._black_female_low_cr._age[-1]
        self.assertAlmostEqual(blackFemaleLowCr, self._black_female_low_cr._gfr)

        whiteMaleHighCr = 141 * (self._white_male_high_cr._creatinine[0]/0.9)**-1.209 * 0.993**self._white_male_high_cr._age[-1]
        self.assertAlmostEqual(whiteMaleHighCr, self._white_male_high_cr._gfr)

        whiteMaleLowCr = 141 * (self._white_male_low_cr._creatinine[0]/0.9)**-.411 * 0.993**self._white_male_low_cr._age[-1]
        self.assertAlmostEqual(whiteMaleLowCr, self._white_male_low_cr._gfr)

if __name__ == "__main__":
    unittest.main()
