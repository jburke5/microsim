from mcm.person import Person
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.outcome_model_repository import OutcomeModelRepository
from mcm.cv_outcome_determination import CVOutcomeDetermination
from mcm.outcome import Outcome
from mcm.outcome import OutcomeType
from mcm.education import Education

from mcm.smoking_status import SmokingStatus
import unittest
import copy


def initializeAFib(person):
    return None


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
            45,
            0,
            Education.COLLEGEGRADUATE,
            SmokingStatus.NEVER,
            initializeAFib)
        self._always_positive_repository = AlwaysPositiveOutcomeRepository()
        self._always_negative_repository = AlwaysNegativeOutcomeRepository()
        self.cvDeterminer = CVOutcomeDetermination(self._always_positive_repository)

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

    def test_will_have_fatal_mi(self):
        self.assertEqual(self.cvDeterminer._will_have_fatal_mi(self.joe, 1.0), 1)
        self.assertEqual(self.cvDeterminer._will_have_fatal_mi(self.joe, 0.0), 0)

    def test_fatal_mi_secondary_prob(self):
        self.cvDeterminer.mi_secondary_case_fatality = 1.0
        self.cvDeterminer.mi_case_fatality = 0.0

        joeClone = copy.deepcopy(self.joe)

        self.assertEqual(self.cvDeterminer._will_have_fatal_mi(joeClone, 0.0), 0)

        joeClone._outcomes[OutcomeType.MI] = (joeClone._age, Outcome(OutcomeType.MI, False))
        # even though the passed fatality rate is zero, it shoudl be overriden by the
        # secondary rate given that joeclone had a prior MI
        self.assertEqual(self.cvDeterminer._will_have_fatal_mi(joeClone, 0.0), 1)

    def test_fatal_stroke_secondary_prob(self):
        self.cvDeterminer.stroke_secondary_case_fatality = 1.0
        self.cvDeterminer.stroke_case_fatality = 0.0

        joeClone = copy.deepcopy(self.joe)

        self.assertEqual(self.cvDeterminer._will_have_fatal_stroke(joeClone, 0.0), 0)

        joeClone._outcomes[OutcomeType.STROKE] = (
            joeClone._age, Outcome(OutcomeType.STROKE, False))
        # even though the passed fatality rate is zero, it shoudl be overriden by the
        # secondary rate given that joeclone had a prior stroke
        self.assertEqual(self.cvDeterminer._will_have_fatal_stroke(joeClone, 0.0), 1)

    def test_will_have_fatal_stroke(self):
        self.assertEqual(self.cvDeterminer._will_have_fatal_stroke(self.joe, 1.0), 1)
        self.assertEqual(self.cvDeterminer._will_have_fatal_stroke(self.joe, 0.0), 0)

    def test_has_mi_vs_stroke(self):
        self.assertEqual(self.cvDeterminer._will_have_mi(self.joe, None, 1.0), 1)
        self.assertEqual(self.cvDeterminer._will_have_mi(self.joe, None, 0.0), 0)

    def test_advance_outcomes_fatal_mi(self):
        self._always_positive_repository.stroke_case_fatality = 1.0
        self._always_positive_repository.mi_case_fatality = 1.0
        self._always_positive_repository.manualStrokeMIProbability = 1.0

        self.joe.advance_outcomes(self._always_positive_repository)
        self.assertTrue(self.joe.has_mi_during_simulation())
        self.assertFalse(self.joe.has_stroke_during_simulation())
        self.assertTrue(self.joe.is_dead())

    def test_advance_outcomes_fatal_stroke(self):
        self._always_positive_repository.stroke_case_fatality = 1.0
        self._always_positive_repository.mi_case_fatality = 1.0
        self._always_positive_repository.manualStrokeMIProbability = 0.0

        self.joe.advance_outcomes(self._always_positive_repository)
        self.assertFalse(self.joe.has_mi_during_simulation())
        self.assertTrue(self.joe.has_stroke_during_simulation())
        self.assertTrue(self.joe.is_dead())

    def test_advance_outcomes_nonfatal_mi(self):
        self.assertEqual(0, self.joe.is_dead())
        self._always_positive_repository.stroke_case_fatality = 0.0
        self._always_positive_repository.mi_case_fatality = 0.0
        self._always_positive_repository.manualStrokeMIProbability = 1.0

        self.joe.advance_outcomes(self._always_positive_repository)
        self.assertTrue(self.joe.has_mi_during_simulation())
        self.assertFalse(self.joe.has_stroke_during_simulation())

    def test_advance_outcomes_nonfatal_stroke(self):
        self._always_positive_repository.stroke_case_fatality = 0.0
        self._always_positive_repository.mi_case_fatality = 0.0
        self._always_positive_repository.manualStrokeMIProbability = 0.0

        self.joe.advance_outcomes(self._always_positive_repository)
        self.assertFalse(self.joe.has_mi_during_simulation())
        self.assertTrue(self.joe.has_stroke_during_simulation())


if __name__ == "__main__":
    unittest.main()
