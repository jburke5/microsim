import unittest

import pandas as pd

from microsim.person.person_filter import PersonFilter
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.population.population_factory import PopulationFactory
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType


def _filtered_adults(filterName, filterFunction):
    pf = PersonFilterFactory.get_person_filter(["adult"])
    pf.add_filter("df", filterName, filterFunction)
    return pf


class TestApplyPersonFiltersOnDf(unittest.TestCase):
    """The df-level filters run on whichever dataframe the people are built from, which with
       distributions is the dataframe of draws rather than the NHANES rows."""

    def _df(self):
        return pd.DataFrame({"age": [10, 30, 50], "sbp": [110., 130., 150.]})

    def test_filters_are_applied(self):
        pf = PersonFilter()
        pf.add_filter("df", "adult", lambda x: x["age"] >= 18)
        kept = PopulationFactory.apply_person_filters_on_df(pf, self._df())
        self.assertEqual([30, 50], kept["age"].tolist())

    def test_filters_are_combined(self):
        pf = PersonFilter()
        pf.add_filter("df", "adult", lambda x: x["age"] >= 18)
        pf.add_filter("df", "lowSBPLimit", lambda x: x["sbp"] > 140)
        kept = PopulationFactory.apply_person_filters_on_df(pf, self._df())
        self.assertEqual([50], kept["age"].tolist())

    def test_none_keeps_everything(self):
        self.assertEqual(3, PopulationFactory.apply_person_filters_on_df(None, self._df()).shape[0])

    def test_a_filter_after_an_emptying_filter_does_not_raise(self):
        pf = PersonFilter()
        pf.add_filter("df", "nobody", lambda x: x["age"] > 1000)
        pf.add_filter("df", "adult", lambda x: x["age"] >= 18)
        self.assertEqual(0, PopulationFactory.apply_person_filters_on_df(pf, self._df()).shape[0])

    def test_a_missing_variable_is_reported_with_the_filter_name(self):
        pf = PersonFilter()
        pf.add_filter("df", "year1999", lambda x: x["year"] == 1999)
        with self.assertRaises(RuntimeError) as cm:
            PopulationFactory.apply_person_filters_on_df(pf, self._df())
        self.assertIn("year1999", str(cm.exception))
        self.assertIn("year", str(cm.exception))


class TestFiltersHoldForTheDraws(unittest.TestCase):
    """Slow: builds the distributions for a NHANES year. One run, several assertions, because the
       fitting step dominates the runtime."""

    @classmethod
    def setUpClass(cls):
        cls.n = 40
        pf = _filtered_adults("lowSBPLimit", lambda x: x[DynamicRiskFactorsType.SBP.value] > 126)
        pf.add_filter("df", "under60", lambda x: x[DynamicRiskFactorsType.AGE.value] < 60)
        cls.people = PopulationFactory.get_nhanes_people(n=cls.n, year=1999, personFilters=pf,
                                                         distributions=True)

    def test_returns_exactly_n_people(self):
        self.assertEqual(self.n, len(self.people))

    def test_the_continuous_filter_holds_for_every_person(self):
        self.assertGreater(min(person._sbp[0] for person in self.people), 126)

    def test_the_age_filter_holds_after_the_age_has_been_rounded(self):
        #the draws are rounded to whole years before the filters see them, so a draw of 59.6 is a
        #60 year old and has to be rejected, not kept
        self.assertLessEqual(max(person._age[0] for person in self.people), 59)

    def test_the_default_adult_filter_still_holds(self):
        self.assertGreaterEqual(min(person._age[0] for person in self.people), 18)


if __name__ == "__main__":
    unittest.main()
