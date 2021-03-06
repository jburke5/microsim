from microsim.gender import NHANESGender
from microsim.person import Person
from microsim.outcome_model_type import OutcomeModelType
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory

import unittest


def initializeAfib(person):
    return None


class TestOutcomeRepository(unittest.TestCase):

    def setUp(self):
        self._white_male = Person(
            age=55, gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, waist=34, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER, alcohol=AlcoholCategory.NONE, 
            antiHypertensiveCount=0,
            statin=0, otherLipidLoweringMedicationCount=0, initializeAfib=initializeAfib)

        self._black_male = Person(
            age=55, gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=200, ldl=90, trig=150,
            bmi=22, waist=34, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER, alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0, otherLipidLoweringMedicationCount=0, initializeAfib=initializeAfib)

        self._black_treated_male = Person(
            age=55, gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=200, ldl=90, trig=150,
            bmi=22, waist=34, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER, alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=1,
            statin=0, otherLipidLoweringMedicationCount=0, initializeAfib=initializeAfib)

        self._white_female = Person(
            age=55, gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, waist=34, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER, alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0, otherLipidLoweringMedicationCount=0, initializeAfib=initializeAfib)

        self._black_female = Person(
            age=55, gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, waist=34, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER, alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0, otherLipidLoweringMedicationCount=0, initializeAfib=initializeAfib)

        self._outcome_model_repository = OutcomeModelRepository()

    def test_get_model_for_person(self):
        self.assertEqual(0.106501, self._outcome_model_repository.select_model_for_person(
            self._white_female, OutcomeModelType.CARDIOVASCULAR).parameters['lagAge'])
        self.assertEqual(0.106501, self._outcome_model_repository.select_model_for_person(
            self._black_female, OutcomeModelType.CARDIOVASCULAR).parameters['lagAge'])
        self.assertEqual(0.064200, self._outcome_model_repository.select_model_for_person(
            self._white_male, OutcomeModelType.CARDIOVASCULAR).parameters['lagAge'])
        self.assertEqual(0.064200, self._outcome_model_repository.select_model_for_person(
            self._black_male, OutcomeModelType.CARDIOVASCULAR).parameters['lagAge'])

    def test_calculate_risk_for_person(self):
        self.assertAlmostEqual(0.017654, self._outcome_model_repository.get_risk_for_person(
            self._black_female, OutcomeModelType.CARDIOVASCULAR, 10), delta=0.00001)
        # note that the reference value here is the corrected version of the
        # appendis table with the tot_chol/hdl ratio set to 4 for both the overall term and
        # the race interaction term
        self.assertAlmostEqual(.03476, self._outcome_model_repository.get_risk_for_person(
            self._black_male, OutcomeModelType.CARDIOVASCULAR, 10), delta=0.00001)

    # details of risk worked out in example_treated_ascvd_scenario.xlsx
    def test_calculate_risk_for_treated_person(self):
        self.assertAlmostEqual(0.069810753, self._outcome_model_repository.get_risk_for_person(
            self._black_treated_male, OutcomeModelType.CARDIOVASCULAR, 10), delta=0.00001)

    if __name__ == "__main__":
        unittest.main()
