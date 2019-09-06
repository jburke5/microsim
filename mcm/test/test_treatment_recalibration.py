import unittest
import pandas as pd
import numpy as np

from mcm.population import NHANESDirectSamplePopulation
from mcm.outcome_model_repository import OutcomeModelRepository
from mcm.outcome import Outcome, OutcomeType


strokeTarget = 0.79
miTarget = 0.91


class TestCoinFlipStrokeModelRepository(OutcomeModelRepository):
    def __init__(self):
        super().__init__()

    # override base class and always return a stroke event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, True) if np.random.uniform() < 0.5 else None


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


def addABPMedMI(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5.5, '_dbp': -3.1}, {OutcomeType.MI: 0.87}


class TestTreatmentRecalibration(unittest.TestCase):
    def setUp(self):
        self.popSize = 1000

        self.alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        self.alwaysMIPop._outcome_model_repository = TestAlwaysMIModelRepository()

    # ok...key insight...because we're goign to recalibrate people, we need to do it by recalibrating a N
    # that reflects the weigthed average (or someting like that) of their risks
    # when you recalibrate a lot of people, you don't want to recalibrate the "average" risk person,
    # you want to recalibrate either the highest risk untreated people or lowest risk treated  people dependingon
    # whether youre' recalibrating up or down

    # also...i'm, prettuy sure that the existing risk flip is going in the wrong direction...
    # if we are over-estimating the treamtent effect, we need to undo events...and we need to undo them amongst the treated

    # basically...all of this is going to need a lot of work..

    def testRecalibrationReducesNumberOfStrokesWhenTargetEffectSizeIsLessThanBaseline(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestCoinFlipStrokeModelRepository()
        alwaysStrokePop.advance(1)
        # everybody has a stroke until we calibrate events...
        numberOfStrokesInBasePopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestCoinFlipStrokeModelRepository()
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeSmallEffectSize)
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertLess(numberOfStrokesInBasePopulation,
                           numberOfStrokesInRecalibratedPopulation)
