"""A draw pass builds at most MAX_DRAW_BATCH Person-objects. Before the cap, a pass built
shortfall/acceptanceRate of them before the person-level filters ran, which took a 500,000-person
epilepsy trial (~1.3% acceptance) to 363 GB at OSC."""

import unittest
from unittest import mock

from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.population.population_factory import PopulationFactory

CAP = 50


def _age_filter(minAge, dfFilters=("adult",)):
    pf = PersonFilterFactory.get_person_filter(list(dfFilters))
    pf.add_filter("person", "minAge", lambda x: x._age[0] >= minAge)
    return pf


class TestDrawBatchIsBounded(unittest.TestCase):
    def _draw(self, **kwargs):
        """Returns the people; self.built gets the number of Person-objects each pass handed to the
        filters, also when the draw raises."""
        spy = mock.Mock(wraps=PopulationFactory.apply_person_filters_on_people)
        self.built = []
        try:
            with (
                mock.patch.object(PopulationFactory, "MAX_DRAW_BATCH", CAP),
                mock.patch.object(PopulationFactory, "apply_person_filters_on_people", spy),
            ):
                return PopulationFactory.get_nhanes_people(year=1999, **kwargs)
        finally:
            self.built = [c.args[1].shape[0] for c in spy.call_args_list]

    def test_a_pass_never_builds_more_than_the_cap(self):
        # ~5% of adults are 75+, so 20 people take hundreds of draws
        people = self._draw(n=20, personFilters=_age_filter(75), nhanesWeights=True)
        self.assertGreater(len(self.built), 1)
        self.assertLessEqual(max(self.built), CAP)
        self.assertEqual(20, len(people))
        self.assertGreaterEqual(min(p._age[0] for p in people), 75)

    def test_cap_holds_while_nothing_is_accepted(self):
        pf = PersonFilterFactory.get_person_filter(["adult"])
        pf.add_filter("person", "nobody", lambda x: False)
        with self.assertRaises(RuntimeError) as cm:
            self._draw(n=5, personFilters=pf, maxDraws=400)
        self.assertIn("None of the 400 rows", str(cm.exception))
        # the batch doubles while nothing is accepted, the cap has to stop it too
        self.assertLessEqual(max(self.built), CAP)

    def test_cap_holds_with_distributions(self):
        pf = _age_filter(75, dfFilters=("adult", "lowSBPLimit"))
        people = self._draw(n=20, personFilters=pf, nhanesWeights=True, distributions=True)
        self.assertLessEqual(max(self.built), CAP)
        self.assertEqual(20, len(people))
        self.assertGreater(min(p._sbp[0] for p in people), 126)
        self.assertGreaterEqual(min(p._age[0] for p in people), 75)


if __name__ == "__main__":
    unittest.main()
