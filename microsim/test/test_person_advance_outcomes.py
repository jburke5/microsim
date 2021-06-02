from microsim.person import Person
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.cv_outcome_determination import CVOutcomeDetermination
from microsim.outcome import Outcome
from microsim.outcome import OutcomeType
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.smoking_status import SmokingStatus
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe,
)
import unittest
import copy


def initializeAFib(person):
    return None


class AlwaysPositiveOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysPositiveOutcomeRepository, self).__init__()

    # override super to alays return a probability of each outcom eas 1
    def get_risk_for_person(self, person, outcome, years=1, vectorized=False):
        return 1


class AlwaysNegativeOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysNegativeOutcomeRepository, self).__init__()

    # override super to alays return a probability of each outcome as 0
    def get_risk_for_person(self, person, outcome, years=1, vectorized=False):
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
            AlcoholCategory.NONE,
            0,
            0,
            0,
            initializeAFib,
        )

        self.joe_with_mi = copy.deepcopy(self.joe)
        self.joe_with_mi.add_outcome_event(Outcome(OutcomeType.MI, False))

        self.joe_with_stroke = copy.deepcopy(self.joe)
        self.joe_with_stroke.add_outcome_event(Outcome(OutcomeType.STROKE, False))

        self._population_dataframe = init_vectorized_population_dataframe(
            [self.joe, self.joe_with_mi, self.joe_with_stroke],
            with_base_gcp=True,
        )

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
        joe_data = self._population_dataframe.iloc[0]

        is_max_prob_mi_fatal = self.cvDeterminer._will_have_fatal_mi(
            joe_data,
            vectorized=True,
            overrideMIProb=1.0,
        )
        is_min_prob_mi_fatal = self.cvDeterminer._will_have_fatal_mi(
            joe_data,
            vectorized=True,
            overrideMIProb=0.0,
        )

        self.assertTrue(is_max_prob_mi_fatal)
        self.assertFalse(is_min_prob_mi_fatal)

    def test_fatal_mi_secondary_prob(self):
        self.cvDeterminer.mi_secondary_case_fatality = 1.0
        self.cvDeterminer.mi_case_fatality = 0.0
        joe_data = self._population_dataframe.iloc[0]
        joe_with_mi_data = self._population_dataframe.iloc[1]  # same as joe_data plus 1 MI

        will_have_fatal_first_mi = self.cvDeterminer._will_have_fatal_mi(
            joe_data,
            vectorized=True,
            overrideMIProb=0.0,
        )
        will_have_fatal_second_mi = self.cvDeterminer._will_have_fatal_mi(
            joe_with_mi_data,
            vectorized=True,
            overrideMIProb=0.0,
        )

        self.assertFalse(will_have_fatal_first_mi)
        # even though the passed fatality rate is zero, it should be overriden by the
        # secondary rate given that joe had a prior MI
        self.assertTrue(will_have_fatal_second_mi)

    def test_fatal_stroke_secondary_prob(self):
        self.cvDeterminer.stroke_secondary_case_fatality = 1.0
        self.cvDeterminer.stroke_case_fatality = 0.0
        joe_data = self._population_dataframe.iloc[0]
        joe_with_stroke_data = self._population_dataframe.iloc[2]  # same as joe_data plus 1 stroke

        will_have_fatal_first_stroke = self.cvDeterminer._will_have_fatal_stroke(
            joe_data,
            vectorized=True,
            overrideStrokeProb=0.0,
        )
        will_have_fatal_second_stroke = self.cvDeterminer._will_have_fatal_stroke(
            joe_with_stroke_data,
            vectorized=True,
            overrideStrokeProb=0.0,
        )

        self.assertFalse(will_have_fatal_first_stroke)
        # even though the passed fatality rate is zero, it shoudl be overriden by the
        # secondary rate given that joeclone had a prior stroke
        self.assertTrue(will_have_fatal_second_stroke)

    def test_will_have_fatal_stroke(self):
        joe_data = self._population_dataframe.iloc[0]

        is_max_prob_stroke_fatal = self.cvDeterminer._will_have_fatal_stroke(
            joe_data,
            vectorized=True,
            overrideStrokeProb=1.0,
        )
        is_min_prob_stroke_fatal = self.cvDeterminer._will_have_fatal_stroke(
            joe_data,
            vectorized=True,
            overrideStrokeProb=0.0,
        )

        self.assertTrue(is_max_prob_stroke_fatal)
        self.assertFalse(is_min_prob_stroke_fatal)

    def test_has_mi_vs_stroke(self):
        joe_data = self._population_dataframe.iloc[0]

        has_mi_max_manual_prob = self.cvDeterminer._will_have_mi(
            joe_data,
            outcome_model_repository=None,
            vectorized=False,
            manualMIProb=1.0,
        )
        has_mi_min_manual_prob = self.cvDeterminer._will_have_mi(
            joe_data,
            outcome_model_repository=None,
            vectorized=False,
            manualMIProb=0.0,
        )

        self.assertTrue(has_mi_max_manual_prob)
        self.assertFalse(has_mi_min_manual_prob)

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
