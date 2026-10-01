import unittest

from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository
from microsim.population.population_factory import PopulationFactory
from microsim.trials.trial_description import NhanesTrialDescription, KaiserTrialDescription
from microsim.trials.trial import Trial
from microsim.trials.trial_type import TrialType
from microsim.person.person_filter_factory import PersonFilterFactory


def _adults_filter():
    return PersonFilterFactory.get_person_filter(["adult"])


class TestNhanesTrialDescriptionPrevalence(unittest.TestCase):
    def test_peopleargs_contains_outcome_prevalence_model_repository(self):
        desc = NhanesTrialDescription(sampleSize=10, duration=1, year=1999,
                                      personFilters=_adults_filter())
        opmr = desc.peopleArgs.get("outcomePrevalenceModelRepository")
        self.assertIsInstance(opmr, OutcomePrevalenceModelRepository)

    def test_kaiser_trial_description_does_not_include_opmr(self):
        # Kaiser inlines its own prevalence calls in get_kaiser_person and does not
        # use the OutcomePrevalenceModelRepository pathway, so peopleArgs must not
        # carry an OPMR for it.
        desc = KaiserTrialDescription(sampleSize=10, duration=1)
        self.assertNotIn("outcomePrevalenceModelRepository", desc.peopleArgs)


class TestTrialSeedsPrevalentOutcomes(unittest.TestCase):
    def setUp(self):
        self.desc = NhanesTrialDescription(
            trialType=TrialType.COMPLETELY_RANDOMIZED,
            sampleSize=10,
            duration=1,
            treatmentStrategies="noTreatment",
            year=1999,
            nhanesWeights=True,
            personFilters=_adults_filter(),
        )

    def test_trial_does_not_replace_description_opmr_instance(self):
        # The OPMR baked into peopleArgs flows through **peopleArgs to both
        # treated and control populations, so Trial construction must not
        # swap it out.
        opmrBefore = self.desc.peopleArgs["outcomePrevalenceModelRepository"]
        Trial(self.desc)
        self.assertIs(self.desc.peopleArgs["outcomePrevalenceModelRepository"], opmrBefore)

    def test_trial_persons_have_cognition_seeded_in_both_arms(self):
        trial = Trial(self.desc)
        for person in list(trial.treatedPop._people) + list(trial.controlPop._people):
            cognition = person._outcomes[OutcomeType.COGNITION]
            self.assertEqual(1, len(cognition))
            self.assertTrue(cognition[0][1].priorToSim)


class TestGetNhanesPeoplePassThrough(unittest.TestCase):
    def test_none_skips_prevalent_outcome_seeding(self):
        # Strict pass-through: explicit None means no prevalence seeding for the
        # constructed persons.
        people = PopulationFactory.get_nhanes_people(
            n=10, year=1999, personFilters=_adults_filter(), nhanesWeights=True,
            outcomePrevalenceModelRepository=None,
        )
        for person in people:
            self.assertEqual(0, len(person._outcomes[OutcomeType.COGNITION]))

    def test_provided_instance_seeds_prevalent_outcomes(self):
        opmr = OutcomePrevalenceModelRepository()
        people = PopulationFactory.get_nhanes_people(
            n=10, year=1999, personFilters=_adults_filter(), nhanesWeights=True,
            outcomePrevalenceModelRepository=opmr,
        )
        for person in people:
            cognition = person._outcomes[OutcomeType.COGNITION]
            self.assertEqual(1, len(cognition))
            self.assertTrue(cognition[0][1].priorToSim)


class TestKaiserTrialEndToEnd(unittest.TestCase):
    """Kaiser does not use OutcomePrevalenceModelRepository (its prevalence calls are
    inlined in get_kaiser_person). This test pins down that Kaiser trial construction
    still works end-to-end after the OPMR plumbing was added on the NHANES side."""

    def test_kaiser_trial_construction_produces_both_populations(self):
        desc = KaiserTrialDescription(
            trialType=TrialType.COMPLETELY_RANDOMIZED,
            sampleSize=20,
            duration=1,
            treatmentStrategies="noTreatment",
        )
        trial = Trial(desc)
        self.assertEqual(20, len(trial.treatedPop._people))
        self.assertEqual(20, len(trial.controlPop._people))

    def test_kaiser_topup_via_person_filter_does_not_crash(self):
        # Regression: PersonFactory.get_person dispatching to KAISER previously
        # forwarded outcomePrevalenceModelRepository to get_kaiser_person, which
        # has no such kwarg. The crash was dormant whenever the initial draw of n
        # persons survived all filters; it only fires when bring_people_to_target_n
        # actually loops, i.e., a person-level filter drops some draws below n.
        pf = PersonFilterFactory.get_person_filter([])
        pf.add_filter("person", "ageAtLeast60", lambda p: p._age[0] >= 60)
        people = PopulationFactory.get_kaiser_people(n=20, personFilters=pf)
        self.assertEqual(20, people.shape[0])
        for p in people:
            self.assertGreaterEqual(p._age[0], 60)


class TestGetNhanesPopulationDefaultSeeding(unittest.TestCase):
    def test_default_population_has_seeded_prevalent_outcomes(self):
        # get_nhanes_population constructs its own OPMR to preserve the prior
        # default-seeding behavior, even though get_nhanes_people is now
        # strict pass-through.
        pop = PopulationFactory.get_nhanes_population(
            n=10, year=1999, personFilters=_adults_filter(), nhanesWeights=True,
        )
        for person in pop._people:
            cognition = person._outcomes[OutcomeType.COGNITION]
            self.assertEqual(1, len(cognition))
            self.assertTrue(cognition[0][1].priorToSim)


if __name__ == "__main__":
    unittest.main()
