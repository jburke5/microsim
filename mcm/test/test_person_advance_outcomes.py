from mcm.person import Person
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
import unittest


class TestPersonAdvanceOutcomes(unittest.TestCase):

    def setUp(self):
        self.joe = Person(
            42,
            NHANESGender.MALE,
            NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            140,
            90,
            5.5,
            50,
            200,
            25,
            90,
            150,
            SmokingStatus.NEVER)

    def test_dead_is_dead_advance_year(self):
        self.joe._alive[-1] = False
        with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance year"):
            self.joe.advance_year(None)

    def test_dead_is_dead_advance_risk_factors(self):
        self.joe._alive[-1] = False
        with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance risk factors"):
            self.joe.advance_risk_factors(None)

    def test_dead_is_dead_advance_outcomes(self):
        self.joe._alive[-1] = False
        with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance outcomes"):
            self.joe.advance_outcomes()


if __name__ == "__main__":
    unittest.main()
