from mcm.gender import NHANESGender
from mcm.person import Person
from mcm.outcome import Outcome
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.outcome_model_repository import OutcomeModelRepository

import unittest


class TestOutcomeRepository(unittest.TestCase):

    def setUp(self):
        self._white_male = Person(
            age=55, gender=NHANESGender.MALE,
            race_ethnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120, dbp=80, a1c=6, hdl=50, tot_chol=213,
            bmi=22, smoking_status=SmokingStatus.NEVER)

        self._black_male = Person(
            age=55, gender=NHANESGender.MALE,
            race_ethnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120, dbp=80, a1c=6, hdl=50, tot_chol=200,
            bmi=22, smoking_status=SmokingStatus.NEVER)

        self._white_female = Person(
            age=55, gender=NHANESGender.FEMALE,
            race_ethnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=120, dbp=80, a1c=6, hdl=50, tot_chol=213,
            bmi=22, smoking_status=SmokingStatus.NEVER)

        self._black_female = Person(
            age=55, gender=NHANESGender.FEMALE,
            race_ethnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=120, dbp=80, a1c=6, hdl=50, tot_chol=213,
            bmi=22, smoking_status=SmokingStatus.NEVER)

        self._outcome_model_repository = OutcomeModelRepository()

    def test_get_model_for_person(self):
        self.assertEqual(0.106501, self._outcome_model_repository.select_model_for_person(
            self._white_female, Outcome.CARDIOVASCULAR)._age)
        self.assertEqual(0.106501, self._outcome_model_repository.select_model_for_person(
            self._black_female, Outcome.CARDIOVASCULAR)._age)
        self.assertEqual(0.064200, self._outcome_model_repository.select_model_for_person(
            self._white_male, Outcome.CARDIOVASCULAR)._age)
        self.assertEqual(0.064200, self._outcome_model_repository.select_model_for_person(
            self._black_male, Outcome.CARDIOVASCULAR)._age)

    def test_calculate_risk_for_person(self):
        self.assertAlmostEqual(0.017654, self._outcome_model_repository.get_risk_for_person(
            self._black_female, Outcome.CARDIOVASCULAR, 10), delta=0.00001)
        self.assertAlmostEqual(0.033756, self._outcome_model_repository.get_risk_for_person(
            self._black_male, Outcome.CARDIOVASCULAR, 10), delta=0.00001)

    if __name__ == "__main__":
        unittest.main()
