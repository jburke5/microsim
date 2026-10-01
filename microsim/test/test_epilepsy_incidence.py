import unittest
import numpy as np

from microsim.outcomes.epilepsy_model import EpilepsyIncidenceModel
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.education import Education
from microsim.risk_factors.smoking_status import SmokingStatus


class StubPerson:
    '''The minimal person interface EpilepsyIncidenceModel reads.'''
    def __init__(self, waveCompleted):
        self._waveCompleted = waveCompleted
        self._age = [70]
        self._gender = NHANESGender.MALE
        self._raceEthnicity = RaceEthnicity.NON_HISPANIC_WHITE
        self._education = Education.HIGHSCHOOLGRADUATE
        self._smokingStatus = SmokingStatus.NEVER
        self._bmi = [27]
        self._totChol = [190]
        self._ldl = [110]
        self._stroke = False
        self._mi = False
        self._any_antiHypertensive = False
        self._current_ckd = False

    def has_diabetes(self):
        return False

    def has_epilepsy(self):
        return False


class TestEpilepsyIncidence(unittest.TestCase):
    def setUp(self):
        self.model = EpilepsyIncidenceModel()

    def test_risk_is_conditional_one_year_probability(self):
        person = StubPerson(waveCompleted=-1)
        lp = self.model.get_linear_predictor_for_person(person)
        expected = 1. - np.exp(-self.model._cbhfSlope * np.exp(lp))
        self.assertAlmostEqual(expected, self.model.get_risk_for_person(person))

    def test_risk_does_not_grow_with_years_in_simulation(self):
        #with a linear cumulative baseline hazard the conditional annual risk is constant,
        #the old CDF draw grew it linearly with waveCompleted
        risks = [self.model.get_risk_for_person(StubPerson(waveCompleted=w)) for w in [-1, 0, 5, 20]]
        for risk in risks[1:]:
            self.assertAlmostEqual(risks[0], risk)

    def test_prior_epilepsy_persists(self):
        person = StubPerson(waveCompleted=3)
        person.has_epilepsy = lambda: True
        self.assertEqual(1., self.model.get_risk_for_person(person))

    def test_risk_scaling_multiplies(self):
        unscaled = self.model.get_risk_for_person(StubPerson(waveCompleted=2))
        scaled = EpilepsyIncidenceModel(riskScaling=2.).get_risk_for_person(StubPerson(waveCompleted=2))
        self.assertAlmostEqual(2. * unscaled, scaled)


if __name__ == "__main__":
    unittest.main()
