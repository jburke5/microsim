import unittest
import pandas as pd
import numpy as np

from mcm.population import NHANESDirectSamplePopulation
from mcm.outcome_model_repository import OutcomeModelRepository
from mcm.outcome import Outcome, OutcomeType

class TestOftenStrokeModelRepository(OutcomeModelRepository):
    def __init__(self, stroke_rate):
        super().__init__()
        self._stroke_rate = stroke_rate

    # override base class and always return a stroke event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, False) if np.random.random() < self._stroke_rate else None

    def get_risk_for_person(self, person, outcomeModelType, years=1):
        return self._stroke_rate

    def assign_non_cv_mortality(self, person):
        return False


class TestOftenMIModelRepository(OutcomeModelRepository):
    def __init__(self, mi_rate):
        super().__init__()
        self._mi_rate = mi_rate

    # override base class and always return a MI event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.MI, False) if np.random.random() < self._mi_rate else None

    def get_risk_for_person(self, person, outcomeModelType, years=1):
        return self._mi_rate

    def assign_non_cv_mortality(self, person):
        return False


def addABPMedStrokeLargeEffectSize(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.STROKE: 0.5, OutcomeType.MI: 0.92}


def addABPMedStrokeHarm(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.STROKE: 1.5, OutcomeType.MI: 0.92}


def addABPMedMIHarm(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.MI: 1.5, OutcomeType.STROKE: 0.92}


def addABPMedMILargeEffectSize(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.MI: 0.5, OutcomeType.STROKE: 0.92}


class TestTreatmentRecalibration(unittest.TestCase):
    def setUp(self):
        self.popSize = 1000

    # if we specify an effect size that is clinically smaller than the target...
    # then the test should rollback strokes so that we end up with fewer strokes...
    def testRecalibrationIncreasesStrokesWhenEffectSizeIsClincallySmallerButNumericallyLarger(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.advance(1)
        # about half of the people should have a stroke...at baseline
        numberOfStrokesInBasePopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        # on average, treatment will have an RR round 0.95 for the BP lowering effect applied
        # so, we're going to recalibrate to a RR of 1.5...that will lead to many MORE strokes 
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeHarm)
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertLess(numberOfStrokesInBasePopulation,
                          numberOfStrokesInRecalibratedPopulation)

    # if we specivy an effect size that is clinically larger (numerically smaller) than the target...
    # then the test should generate new stroke events so that we end up with more strokes
    def testRecalibrationReducesStrokesWhenEffectSizeIsClincallyLargerButNumericallySmaller(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.advance(1)
        # about half of people shoudl have strokes
        numberOfStrokesInBasePopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeLargeEffectSize)
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertGreater(numberOfStrokesInBasePopulation, numberOfStrokesInRecalibratedPopulation)

    # if we specify an effect size that is clincally smaller than the target...
    # then the test should rollback MIS so that we end up with fewer MIS...
    def testRecalibrationIncreasesSIsWhenEffectSizeIsClincallySmallerButNumericallyLarger(self):
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        alwaysMIPop.advance(1)
        # about half of people have an MI at baseline
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        alwaysMIPop.set_bp_treatment_strategy(addABPMedMIHarm)
        alwaysMIPop.advance(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.iteritems()]).sum()
        self.assertLess(numberOfMIsInBasePopulation,
                        numberOfMIsInRecalibratedPopulation)

    # if we specify an effect size that is larger than the target...
    # then the test should generate new mi events so that we end up with more MIs
    def testRecalibrationReducesMIsWhenEffectSizeIsClincallyLargerButNumericallySmaller(self):
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        neverMIPop.advance(1)
        # abou thalf of hte population has an MI at baseline
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        neverMIPop.set_bp_treatment_strategy(addABPMedMILargeEffectSize)
        neverMIPop.advance(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.iteritems()]).sum()
        self.assertGreater(numberOfMIsInBasePopulation, numberOfMIsInRecalibratedPopulation)
