import unittest
import pandas as pd

from mcm.population import NHANESDirectSamplePopulation
from mcm.outcome_model_repository import OutcomeModelRepository
from mcm.outcome import Outcome, OutcomeType


strokeTarget = 0.79
miTarget = 0.91


class TestAlwaysStrokeModelRepository(OutcomeModelRepository):
    def __init__(self):
        super().__init__()

    # override base class and always return a stroke event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, True)


class TestNeverEventModelRepository(OutcomeModelRepository):
    def __init__(self):
        super().__init__()

    # override base class and always return a stroke event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return None


class TestAlwaysMIModelRepository(OutcomeModelRepository):
    def __init__(self):
        super().__init__()

    # override base class and always return a MI event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.MI, True)


def addABPMedStrokeBaseEffectSize(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.STROKE: strokeTarget}


def addABPMedStrokeSmallEffectSize(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.STROKE: 0.99}

def addABPMedMISmallEffectSize(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.MI: 0.99}


def addABPMedMIBaseEffectSize(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.MI: 0.87}


class TestTreatmentRecalibration(unittest.TestCase):
    def setUp(self):
        self.popSize = 1000

    # if everybody always has a stroke...
    # and we specify an effect size that is smaller than the target...
    # then the test should rollback strokes so that we end up with fewer strokes...
    def testRecalibrationReducesNumberOfStrokesWhenTargetEffectSizeIsLessThanBaseline(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestAlwaysStrokeModelRepository()
        alwaysStrokePop.advance(1)
        # everybody has a stroke until we calibrate events...
        numberOfStrokesInBasePopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestAlwaysStrokeModelRepository()
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeSmallEffectSize)
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertGreater(numberOfStrokesInBasePopulation,
                           numberOfStrokesInRecalibratedPopulation)

    # if nobody ever has a stroke...
    # and we specify an effect size that is larger than the target...
    # then the test should generate new stroke events so that we end up with more strokes
    def testRecalibrationIncreasesNumberOfStrokesWhenTargetEffectSizeIsLargerThanBaseline(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestNeverEventModelRepository()
        alwaysStrokePop.advance(1)
        # everybody has a stroke until we calibrate events...
        numberOfStrokesInBasePopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestNeverEventModelRepository()
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeBaseEffectSize)
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertLess(numberOfStrokesInBasePopulation, numberOfStrokesInRecalibratedPopulation)

    # if everybody always has an MI...
    # and we specify an effect size that is smaller than the target...
    # then the test should rollback strokes so that we end up with fewer MIS...

    def testRecalibrationReducesNumberOfMIsWhenTargetEffectSizeIsLessThanBaseline(self):
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestAlwaysMIModelRepository()
        alwaysMIPop.advance(1)
        # everybody has a stroke until we calibrate events...
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestAlwaysMIModelRepository()
        alwaysMIPop.set_bp_treatment_strategy(addABPMedMISmallEffectSize)
        alwaysMIPop.advance(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.iteritems()]).sum()
        self.assertGreater(numberOfMIsInBasePopulation,
                           numberOfMIsInRecalibratedPopulation)

    # if nobody ever has an  MI...
    # and we specify an effect size that is larger than the target...
    # then the test should generate new stroke events so that we end up with more MIs
    def testRecalibrationIncreasesNumberOfMIsWhenTargetEffectSizeIsLargerThanBaseline(self):
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestNeverEventModelRepository()
        neverMIPop.advance(1)
        # everybody has a stroke until we calibrate events...
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestNeverEventModelRepository()
        neverMIPop.set_bp_treatment_strategy(addABPMedMIBaseEffectSize)
        neverMIPop.advance(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.iteritems()]).sum()
        self.assertLess(numberOfMIsInBasePopulation, numberOfMIsInRecalibratedPopulation)
