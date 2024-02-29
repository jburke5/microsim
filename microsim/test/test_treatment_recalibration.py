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
        return (
            Outcome(OutcomeType.STROKE, False) if np.random.random() < self._stroke_rate else None
        )

    def assign_cv_outcome_vectorized(self, x, rng=None):
        if np.random.random() < self._stroke_rate:
            x.miNext = False
            x.strokeNext = True
            x.deadNext = False
            x.ageAtFirstStroke = (
                x.age
                if (x.ageAtFirstStroke is None) or (np.isnan(x.ageAtFirstStroke))
                else x.ageAtFirstStroke
            )
        else:
            x.miNext = False
            x.strokeNext = False
            x.deadNext = False
        return x

    def get_risk_for_person(self, person, outcomeModelType, years=1, vectorized=False, rng=None):
        return self._stroke_rate

    def assign_non_cv_mortality(self, person):
        return False


class TestOftenMIModelRepository(OutcomeModelRepository):
    def __init__(self, mi_rate, fatality_rate=0.0, non_cv_mortality_rate=0.0):
        super().__init__()
        self._mi_rate = mi_rate
        self._fatality_rate = fatality_rate
        self._non_cv_mortality_rate = non_cv_mortality_rate

    # override base class and always return a MI event
    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return (
            Outcome(OutcomeType.MI, np.random.random() < self._fatality_rate)
            if np.random.random() < self._mi_rate
            else None
        )

    def assign_cv_outcome_vectorized(self, x, rng=None):
        if np.random.random() < self._mi_rate:
            x.miNext = True
            x.strokeNext = False
            x.deadNext = np.random.random() < self._fatality_rate
            x.miFatal = x.deadNext
            x.ageAtFirstMI = (
                x.age if (x.ageAtFirstMI is None) or (np.isnan(x.ageAtFirstMI)) else x.ageAtFirstMI
            )
        else:
            x.miNext = False
            x.strokeNext = False
            x.deadNext = False
        return x

    def get_risk_for_person(self, person, outcomeModelType, years=1, vectorized=False, rng=None):
        return self._mi_rate

    def assign_non_cv_mortality(self, person):
        return np.random.uniform(size=1)[0] < self._non_cv_mortality_rate

    def assign_non_cv_mortality_vectorized(self, person, years=1, rng=None):
        return np.random.uniform(size=1)[0] < self._non_cv_mortality_rate


# Can't inherit from BaseTreatmentStrategy/AddASingleBPMedTreatmentStrategy:
# ABCs and derived classes are not `pickle`-able, which breaks multiprocess/pandarellel
class addABPMedStrokeLargeEffectSize:
    def __init__(self):
        self._sbp_lowering = 5.5
        self._dbp_lowering = 3.1

    def get_changes_for_person(self, person):
        return (
            {"_antiHypertensiveCount": 1},
            {"_bpMedsAdded": 1},
            {"_sbp": -1 * self._sbp_lowering, "_dbp": -1 * self._dbp_lowering},
        )

    def get_treatment_recalibration_for_population(self):
        return {OutcomeType.STROKE: 0.5, OutcomeType.MI: 0.92}

    def get_treatment_recalibration_for_person(self, person):
        return {OutcomeType.STROKE: 0.5, OutcomeType.MI: 0.92}

    def repeat_treatment_strategy(self):
        return False

    def get_changes_vectorized(self, x):
        x.bpMedsAddedNext = 1
        x.totalBPMedsAddedNext = 1
        x.sbpNext = x.sbpNext - self._sbp_lowering
        x.dbpNext = x.dbpNext - self._dbp_lowering
        return x

    def rollback_changes_vectorized(self, x):
        x.sbpNext = x.sbpNext + self._sbp_lowering
        x.dbpNext = x.dbpNext + self._dbp_lowering
        x.bpMedsAddedNext = 0
        x.totalBPMedsAddedNext = 0
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
        self.popSize = 500

    # if we specify an effect size that is clinically smaller than the target...
    # then the test should rollback strokes so that we end up with fewer strokes...
    def testRecalibrationIncreasesStrokesWhenEffectSizeIsClincallySmallerButNumericallyLarger(
        self,
    ):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.advance(1)
        # about half of the people should have a stroke...at baseline
        numberOfStrokesInBasePopulation = pd.Series(
            [
                person.has_stroke_during_simulation()
                for i, person in alwaysStrokePop._people.items()
            ]
        ).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        # on average, treatment will have an RR round 0.95 for the BP lowering effect applied
        # so, we're going to recalibrate to a RR of 1.5...that will lead to many MORE strokes
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeHarm())
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [
                person.has_stroke_during_simulation()
                for i, person in alwaysStrokePop._people.items()
            ]
        ).sum()
        self.assertLess(numberOfStrokesInBasePopulation, numberOfStrokesInRecalibratedPopulation)

    # if we specivy an effect size that is clinically larger (numerically smaller) than the target...
    # then the test should generate new stroke events so that we end up with more strokes
    def testRecalibrationReducesStrokesWhenEffectSizeIsClincallyLargerButNumericallySmaller(self):
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.advance(1)
        # about half of people shoudl have strokes
        numberOfStrokesInBasePopulation = pd.Series(
            [
                person.has_stroke_during_simulation()
                for i, person in alwaysStrokePop._people.items()
            ]
        ).sum()

        # set a treatment strategy on teh population
        alwaysStrokePop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysStrokePop._outcome_model_repository = TestOftenStrokeModelRepository(0.5)
        alwaysStrokePop.set_bp_treatment_strategy(addABPMedStrokeLargeEffectSize())
        alwaysStrokePop.advance(1)
        numberOfStrokesInRecalibratedPopulation = pd.Series(
            [
                person.has_stroke_during_simulation()
                for i, person in alwaysStrokePop._people.items()
            ]
        ).sum()

        self.assertGreater(
            numberOfStrokesInBasePopulation, numberOfStrokesInRecalibratedPopulation
        )

    # if we specify an effect size that is clincally smaller than the target...
    # then the test should rollback MIS so that we end up with fewer MIS...
    def testRecalibrationIncreasesSIsWhenEffectSizeIsClincallySmallerButNumericallyLarger(self):
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        alwaysMIPop.advance(1)
        # about half of people have an MI at baseline
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.items()]
        ).sum()

        # set a treatment strategy on teh population
        alwaysMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        alwaysMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        alwaysMIPop.set_bp_treatment_strategy(addABPMedMIHarm())
        alwaysMIPop.advance(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in alwaysMIPop._people.items()]
        ).sum()

        self.assertLess(numberOfMIsInBasePopulation, numberOfMIsInRecalibratedPopulation)

    # if we specify an effect size that is larger than the target...
    # then the test should generate new mi events so that we end up with more MIs
    def testRecalibrationReducesMIsWhenEffectSizeIsClincallyLargerButNumericallySmaller(self):
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        neverMIPop.advance(1)
        # abou thalf of hte population has an MI at baseline
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.items()]
        ).sum()

        # set a treatment strategy on teh population
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(0.5)
        neverMIPop.set_bp_treatment_strategy(addABPMedMILargeEffectSize())
        neverMIPop.advance(1)
        numberOfMIsInRecalibratedPopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.items()]
        ).sum()

        self.assertGreater(numberOfMIsInBasePopulation, numberOfMIsInRecalibratedPopulation)

    def testRollbackFatalEventsRollsBackDeath(self):
        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(1.0, 1.0)
        neverMIPop.advance(1)
        # the whole popuulation should have MIs at baseline
        numberOfMIsInBasePopulation = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.items()]
        ).sum()
        self.assertEqual(self.popSize, numberOfMIsInBasePopulation)
        numberOfFatalMIsInBasePopulation = pd.Series(
            [
                person.has_mi_during_simulation() & person.is_dead()
                for i, person in neverMIPop._people.items()
            ]
        ).sum()
        self.assertEqual(self.popSize, numberOfFatalMIsInBasePopulation)

        neverMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        neverMIPop._outcome_model_repository = TestOftenMIModelRepository(1.0, 1.0)
        # this requires that we rollback a lot of events.
        neverMIPop.set_bp_treatment_strategy(addABPMedMILargeEffectSize())
        neverMIPop.advance(1)

        numberOfMIsAfterRecalibration = pd.Series(
            [person.has_mi_during_simulation() for i, person in neverMIPop._people.items()]
        ).sum()
        numberOfFatalMIsAfterRecalibration = pd.Series(
            [
                person.has_mi_during_simulation() & person.is_dead()
                for i, person in neverMIPop._people.items()
            ]
        ).sum()

        self.assertGreater(numberOfFatalMIsInBasePopulation, numberOfFatalMIsAfterRecalibration)

    def testAdvanceAfterRollbackWorksOnWholePopulation(self):
        oftenMIPop = NHANESDirectSamplePopulation(self.popSize, 2001)
        oftenMIPop._outcome_model_repository = TestOftenMIModelRepository(0.2, 0.2, 0.2)
        # this requires that we rollback a lot of events.
        oftenMIPop.set_bp_treatment_strategy(addABPMedMILargeEffectSize())
        oftenMIPop.advance(5)

        ageLength = pd.Series([len(person._age) for i, person in oftenMIPop._people.items()])
        dead = pd.Series([person.is_dead() for i, person in oftenMIPop._people.items()])

        numberWithFullFollowup = pd.Series(
            [
                person.is_dead() or len(person._age) == 6
                for i, person in oftenMIPop._people.items()
            ]
        ).sum()
        # some people were getting "lost" when they had events to rollback of if the had non CV daeths...
        # this way everybody either is clearly marekd as dead or has compelte follow up
        self.assertEqual(self.popSize, numberWithFullFollowup)


if __name__ == "__main__":
    unittest.main()
