import itertools
import unittest
from collections import Counter

from microsim.outcomes.outcome import OutcomeType
from microsim.population.population_factory import PopulationFactory
from microsim.trials.trial import Trial
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory

#characterization tests for the rate-pair (Option C) migration:
#every expected value is recomputed from Person-level primitives, which the migration does not touch,
#so these tests must pass unchanged before and after the migration


def followup_rate_from_person_primitives(people, outcomesTypeList, wave):
    events = sum(p.has_any_outcome_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave) for p in people)
    personYears = sum(p.get_followup_person_years_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave) for p in people)
    return 1000. * events / personYears


def raw_incidence_from_person_primitives(pop, outcomeType, groups=False):
    outcomeAges = [p.get_first_incidence_age(outcomeType) for p in pop._people]
    outcomeAgesCounter = Counter(a for a in outcomeAges if a is not None)
    atRiskAges = itertools.chain.from_iterable(p.get_first_incidence_at_risk_ages(outcomeType) for p in pop._people)
    agesCounter = Counter(atRiskAges)
    if groups:
        outcomeAgesCounter = pop.get_ages_group_counter(outcomeAgesCounter)
        agesCounter = pop.get_ages_group_counter(agesCounter)
    return {age: (outcomeAgesCounter[age]/agesCounter[age] if (age in outcomeAgesCounter and agesCounter[age] != 0) else 0)
            for age in agesCounter.keys()}


class TestPopulationRateFunctions(unittest.TestCase):
    '''Pins get_outcome_incidence_rates_at_end_of_wave and get_raw_incidence_by_age against Person primitives.'''

    @classmethod
    def setUpClass(cls):
        cls.pop = PopulationFactory.get_nhanes_population(n=200, year=1999, nhanesWeights=True)
        cls.pop.advance(3)

    def test_incidence_rates_at_end_of_wave(self):
        for outcomesTypeList in [[OutcomeType.DEATH], [OutcomeType.STROKE], [OutcomeType.STROKE, OutcomeType.MI, OutcomeType.DEATH]]:
            for wave in [0, 2]:
                expected = followup_rate_from_person_primitives(self.pop._people, outcomesTypeList, wave)
                actual = self.pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave)
                self.assertAlmostEqual(expected, actual, places=10)

    def test_raw_incidence_by_age(self):
        for outcomeType in [OutcomeType.STROKE, OutcomeType.DEMENTIA, OutcomeType.DEATH]:
            for groups in [False, True]:
                expected = raw_incidence_from_person_primitives(self.pop, outcomeType, groups=groups)
                actual = self.pop.get_raw_incidence_by_age(outcomeType, groups=groups)
                self.assertEqual(expected, actual)


class TestKaiserScdModalityRates(unittest.TestCase):
    '''Pins get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave against Person primitives.'''

    @classmethod
    def setUpClass(cls):
        cls.pop = PopulationFactory.get_kaiser_population(n=100)
        cls.pop.advance(2)

    def test_rates_by_scd_and_modality(self):
        outcomesTypeList = [OutcomeType.STROKE, OutcomeType.DEMENTIA]
        wave = 1
        eventsForGroup = Counter()
        personYearsForGroup = Counter()
        for group, p in zip(self.pop.get_scd_by_modality_group(), self.pop._people):
            eventsForGroup[group] += p.has_any_outcome_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave)
            personYearsForGroup[group] += p.get_followup_person_years_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave)
        expected = {group: 1000. * eventsForGroup[group] / personYearsForGroup[group]
                    for group in sorted(eventsForGroup)}
        actual = self.pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave)
        self.assertEqual(set(expected.keys()), set(actual.keys()))
        for group in expected:
            self.assertAlmostEqual(expected[group], actual[group], places=10)


class TestTrialIncidenceRateAssessments(unittest.TestCase):
    '''Pins the six factory IR assessments end-to-end: whatever interface the assessor uses internally,
       the reported rates must equal the Person-primitive computation on each trial arm.'''

    IR_ASSESSMENTS = {"strokeIR": [OutcomeType.STROKE],
                      "miIR": [OutcomeType.MI],
                      "deathIR": [OutcomeType.DEATH],
                      "dementiaIR": [OutcomeType.DEMENTIA],
                      "mciIR": [OutcomeType.MCI],
                      "strokeOrDementiaOrMciIR": [OutcomeType.STROKE, OutcomeType.DEMENTIA, OutcomeType.MCI]}

    @classmethod
    def setUpClass(cls):
        description = NhanesTrialDescription(sampleSize=100, duration=3, treatmentStrategies="1bpMedsAdded")
        cls.trial = Trial(description)
        cls.trial.run_analyze(TrialOutcomeAssessorFactory.get_trial_outcome_assessor(), notify=False)

    def test_ir_results_match_person_primitives(self):
        results = self.trial.results["incidenceRate"]
        for name, outcomesTypeList in self.IR_ASSESSMENTS.items():
            self.assertIn(name, results)
            treatedRate, controlRate = results[name]
            for rate, pop in [(treatedRate, self.trial.treatedPop), (controlRate, self.trial.controlPop)]:
                expected = followup_rate_from_person_primitives(pop._people, outcomesTypeList, pop._waveCompleted)
                self.assertAlmostEqual(expected, rate, places=10)


if __name__ == "__main__":
    unittest.main()
