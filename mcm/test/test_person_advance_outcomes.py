from mcm.person import Person
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.outcome_model_repository import OutcomeModelRepository

from mcm.smoking_status import SmokingStatus
import unittest


class AlwaysPositiveOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysPositiveOutcomeRepository, self).__init__()

    # override super to alays return a probability of each outcom eas 1
    def get_risk_for_person(self, person, outcome, years=1):
        return 1


class AlwaysNegativeOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysNegativeOutcomeRepository, self).__init__()

    # override super to alays return a probability of each outcome as 0
    def get_risk_for_person(self, person, outcome, years=1):
        return 0


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
        self._always_positive_repository = AlwaysPositiveOutcomeRepository()
        self._always_negative_repository = AlwaysNegativeOutcomeRepository()

    def test_dead_is_dead_advance_year(self):
        self.joe._alive[-1] = False
        with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance year"):
            self.joe.advance_year(None, None)

    def test_dead_is_dead_advance_risk_factors(self):
        self.joe._alive[-1] = False
        with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance risk factors"):
            self.joe.advance_risk_factors(None)

    def test_dead_is_dead_advance_outcomes(self):
        self.joe._alive[-1] = False
        with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance outcomes"):
            self.joe.advance_outcomes(None)

    def test_has_fatal_mi(self):
        self.assertEqual(self.joe._has_fatal_mi(1.0), 1)
        self.assertEqual(self.joe._has_fatal_mi(0.0), 0)

    def test_has_fatal_stroke(self):
        self.assertEqual(self.joe._has_fatal_stroke(1.0), 1)
        self.assertEqual(self.joe._has_fatal_stroke(0.0), 0)

    def test_has_mi_vs_stroke(self):
        self.assertEqual(self.joe._has_mi(1.0), 1)
        self.assertEqual(self.joe._has_mi(0.0), 0)

    def test_advance_outcomes_fatal_mi(self):
        self.joe.advance_outcomes(
            self._always_positive_repository,
            miVsStrokeProbability=1.0,
            fatalMIPRob=1.0,
            fatalStrokeProb=1.0)
        self.assertEqual(1, self.joe._mi)
        self.assertEqual(0, self.joe._stroke)
        self.assertEqual(1, self.joe.is_dead())

    def test_advance_outcomes_fatal_stroke(self):
        self.joe.advance_outcomes(
            self._always_positive_repository,
            miVsStrokeProbability=0.0,
            fatalMIPRob=1.0,
            fatalStrokeProb=1.0)
        self.assertEqual(0, self.joe._mi)
        self.assertEqual(1, self.joe._stroke)
        self.assertEqual(1, self.joe.is_dead())

    def test_advance_outcomes_nonfatal_mi(self):
        self.assertEqual(0, self.joe.is_dead())
        self.joe.advance_outcomes(
            self._always_positive_repository,
            miVsStrokeProbability=1.0,
            fatalMIPRob=0.0,
            fatalStrokeProb=1.0)
        self.assertEqual(1, self.joe._mi)
        self.assertEqual(0, self.joe._stroke)

    def test_advance_outcomes_nonfatal_stroke(self):
        self.joe.advance_outcomes(
            self._always_positive_repository,
            miVsStrokeProbability=0.0,
            fatalMIPRob=0.0,
            fatalStrokeProb=0.0)
        self.assertEqual(0, self.joe._mi)
        self.assertEqual(1, self.joe._stroke)


if __name__ == "__main__":
    unittest.main()
