from mcm.gender import NHANESGender
from mcm.person import Person
from mcm.outcome_model_type import OutcomeModelType
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.outcome_model_repository import OutcomeModelRepository

import unittest


class TestOutcomeRepository(unittest.TestCase):

    def setUp(self):
        self._white_male = Person(
            age=55, gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, smokingStatus=SmokingStatus.NEVER)

        self._black_male = Person(
            age=55, gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=200, ldl=90, trig=150,
            bmi=22, smokingStatus=SmokingStatus.NEVER)

        self._white_female = Person(
            age=55, gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, smokingStatus=SmokingStatus.NEVER)

        self._black_female = Person(
            age=55, gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120, dbp=80, a1c=6, hdl=50, totChol=213, ldl=90, trig=150,
            bmi=22, smokingStatus=SmokingStatus.NEVER)

        self._outcome_model_repository = OutcomeModelRepository()

    def test_get_model_for_person(self):
        self.assertEqual(0.106501, self._outcome_model_repository.select_model_for_person(
            self._white_female, OutcomeModelType.CARDIOVASCULAR)._age)
        self.assertEqual(0.106501, self._outcome_model_repository.select_model_for_person(
            self._black_female, OutcomeModelType.CARDIOVASCULAR)._age)
        self.assertEqual(0.064200, self._outcome_model_repository.select_model_for_person(
            self._white_male, OutcomeModelType.CARDIOVASCULAR)._age)
        self.assertEqual(0.064200, self._outcome_model_repository.select_model_for_person(
            self._black_male, OutcomeModelType.CARDIOVASCULAR)._age)

    def test_calculate_risk_for_person(self):
        self.assertAlmostEqual(0.017654, self._outcome_model_repository.get_risk_for_person(
            self._black_female, OutcomeModelType.CARDIOVASCULAR, 10), delta=0.00001)
        # note that the reference value here is the corrected version of the
        # appendis table with the tot_chol/hdl ratio set to 4 for both the overall term and
        # the race interaction term
        self.assertAlmostEqual(.03476, self._outcome_model_repository.get_risk_for_person(
            self._black_male, OutcomeModelType.CARDIOVASCULAR, 10), delta=0.00001)

    if __name__ == "__main__":
        unittest.main()
