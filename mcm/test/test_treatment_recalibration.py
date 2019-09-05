import unittest
import pandas as pd

from mcm.population import NHANESDirectSamplePopulation
from mcm.outcome_model_repository import OutcomeModelRepository
from mcm.outcome import Outcome, OutcomeType


class TestAlwaysStrokeModelRepository(OutcomeModelRepository):
    def __init__(self):
        super().__init__()

    # override base class and always return a stroke event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, True)


class TestAlwaysMIModelRepository(OutcomeModelRepository):
    def __init__(self):
        super().__init__()

    # override base class and always return a MI event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.MI, True)


def addABPMed(person):
    return {'_antiHypertensiveCount': 1}, {'_sbp': - 5, '_dbp': -3}, {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}


class TestTreatmentRecalibration(unittest.TestCase):
    def setUp(self):
        self.popSize = 100

        self.alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        self.alwaysMIPop._outcome_model_repository = TestAlwaysMIModelRepository()

    def testRecalibrationReducesEvents(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestAlwaysStrokeModelRepository()
        alwaysStrokePop.advance(1)
        # everybody has a stroke until we calibrate events...
        numberOfStrokesInPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertEqual(self.popSize, numberOfStrokesInPopulation)
        print("###########")
        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestAlwaysStrokeModelRepository()
        alwaysStrokePop.set_bp_treatment_strategy(addABPMed)
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertGreater(self.popSize, numberOfStrokesInRecalibratedPopulation)
