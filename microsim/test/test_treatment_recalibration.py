import unittest
import pandas as pd
import numpy as np

from microsim.population import NHANESDirectSamplePopulation
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome import Outcome, OutcomeType


class TestOftenStrokeModelRepository(OutcomeModelRepository):
    def __init__(self, stroke_rate):
        super().__init__()
        self._stroke_rate = stroke_rate

    # override base class and always return a stroke event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, False) if np.random.random() < self._stroke_rate else None

    def assign_cv_outcome_vectorized(self, x):
        if np.random.random() < self._stroke_rate:
            x.miNext = False
            x.strokeNext = True
            x.deadNext = False
            x.ageAtFirstStroke = x.age if (x.ageAtFirstStroke is None) or (np.isnan(x.ageAtFirstStroke)) else x.ageAtFirstStroke
        else:
            x.miNext = False
            x.strokeNext = False
            x.deadNext = False
        return x

    def get_risk_for_person(self, person, outcomeModelType, years=1, vectorized=False):
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

    def assign_cv_outcome_vectorized(self, x):
        if np.random.random() < self._mi_rate:
            x.miNext = True
            x.strokeNext = False
            x.deadNext = False
            x.ageAtFirstMI = x.age if (x.ageAtFirstMI is None) or (np.isnan(x.ageAtFirstMI)) else x.ageAtFirstMI
        else:
            x.miNext = False
            x.strokeNext = False
            x.deadNext = False
        return x

    def get_risk_for_person(self, person, outcomeModelType, years=1, vectorized=False):
        return self._mi_rate

    def assign_non_cv_mortality(self, person):
        return False


# Can't inherit from BaseTreatmentStrategy/AddASingleBPMedTreatmentStrategy:
# ABCs and derived classes are not `pickle`-able, which breaks multiprocess/pandarellel
class addABPMedStrokeLargeEffectSize:
    def __init__(self):
        self._sbp_lowering = 5.5
        self._dbp_lowering = 3.1

    def get_changes_for_person(self, person):
        return (
            {'_antiHypertensiveCount': 1},
            {'_bpMedsAdded': 1},
            {'_sbp': -1 * self._sbp_lowering, '_dbp': -1 * self._dbp_lowering},
        )

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.5, OutcomeType.MI: 0.92}

    def get_treatment_recalibration_for_person(self, person):
        return {OutcomeType.STROKE: 0.5, OutcomeType.MI: 0.92}

    def repeat_treatment_strategy(self):
        return False

    def get_changes_vectorized(self, x):
        x.antiHypertensiveCountNext = x.antiHypertensiveCountNext + 1
        x.bpMedsAddedNext = 1
        x.sbpNext = x.sbpNext - self._sbp_lowering
        x.dbpNext = x.dbpNext - self._dbp_lowering
        return x


class addABPMedStrokeHarm(addABPMedStrokeLargeEffectSize):
    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 1.5, OutcomeType.MI: 0.92}

    def get_treatment_recalibration_for_person(self, person):
        return {OutcomeType.STROKE: 1.5, OutcomeType.MI: 0.92}

class addABPMedMIHarm(addABPMedStrokeLargeEffectSize):
    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.MI: 1.5, OutcomeType.STROKE: 0.92}

    def get_treatment_recalibration_for_person(self, person):
        return {OutcomeType.MI: 1.5, OutcomeType.STROKE: 0.92}

class addABPMedMILargeEffectSize(addABPMedStrokeLargeEffectSize):
    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.MI: 0.5, OutcomeType.STROKE: 0.92}

    def get_treatment_recalibration_for_person(self, person):
        return {OutcomeType.MI: 0.5, OutcomeType.STROKE: 0.92}


class TestTreatmentRecalibration(unittest.TestCase):
    def setUp(self):
        self.popSize = 1000

    # if we specify an effect size that is clinically smaller than the target...
    # then the test should rollback strokes so that we end up with fewer strokes...
    def testRecalibrationIncreasesStrokesWhenEffectSizeIsClincallySmallerButNumericallyLarger(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.advance_vectorized(1)
        # about half of the people should have a stroke...at baseline
        numberOfStrokesInBasePopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        # on average, treatment will have an RR round 0.95 for the BP lowering effect applied
        # so, we're going to recalibrate to a RR of 1.5...that will lead to many MORE strokes 
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeHarm())
        alwaysStrokePop.advance_vectorized(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertLess(numberOfStrokesInBasePopulation,
                          numberOfStrokesInRecalibratedPopulation)

    # if we specivy an effect size that is clinically larger (numerically smaller) than the target...
    # then the test should generate new stroke events so that we end up with more strokes
    def testRecalibrationReducesStrokesWhenEffectSizeIsClincallyLargerButNumericallySmaller(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.advance_vectorized(1)
        # about half of people shoudl have strokes
        numberOfStrokesInBasePopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeLargeEffectSize())
        alwaysStrokePop.advance_vectorized(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [person.has_stroke_during_simulation() for i, person in alwaysStrokePop._people.iteritems()]).sum()
        self.assertGreater(numberOfStrokesInBasePopulation, numberOfStrokesInRecalibratedPopulation)

    # if we specify an effect size that is clincally smaller than the target...
    # then the test should rollback MIS so that we end up with fewer MIS...
    def testRecalibrationIncreasesSIsWhenEffectSizeIsClincallySmallerButNumericallyLarger(self):
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        alwaysMIPop.advance_vectorized(1)
        # about half of people have an MI at baseline
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        alwaysMIPop.set_bp_treatment_strategy(addABPMedMIHarm())
        alwaysMIPop.advance_vectorized(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.iteritems()]).sum()
        self.assertLess(numberOfMIsInBasePopulation,
                        numberOfMIsInRecalibratedPopulation)

    # if we specify an effect size that is larger than the target...
    # then the test should generate new mi events so that we end up with more MIs
    def testRecalibrationReducesMIsWhenEffectSizeIsClincallyLargerButNumericallySmaller(self):
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        neverMIPop.advance_vectorized(1)
        # abou thalf of hte population has an MI at baseline
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.iteritems()]).sum()

        # set a treatment strategy on teh population
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        neverMIPop.set_bp_treatment_strategy(addABPMedMILargeEffectSize())
        neverMIPop.advance_vectorized(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.iteritems()]).sum()
        self.assertGreater(numberOfMIsInBasePopulation, numberOfMIsInRecalibratedPopulation)

if __name__ == "__main__":
    unittest.main()