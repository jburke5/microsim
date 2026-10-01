import unittest

import numpy as np
import pandas as pd

from microsim.common.population_type import PopulationType
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.population.population_factory import PopulationFactory
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType


def _adults_filter():
    return PersonFilterFactory.get_person_filter(["adult"])


def _adults_filter_with_person_filter(name, filterFunction):
    """Adult df filter plus a person-level filter, which is what forces bring_people_to_target_n
       to run: person-level filters can only drop people after they have been built."""
    pf = PersonFilterFactory.get_person_filter(["adult"])
    pf.add_filter("person", name, filterFunction)
    return pf


class TestNIsHonored(unittest.TestCase):
    """n used to be silently ignored unless a weighting mode was requested."""

    def test_n_honored_without_any_weights(self):
        people = PopulationFactory.get_nhanes_people(
            n=25, year=1999, personFilters=_adults_filter(), nhanesWeights=False,
        )
        self.assertEqual(25, people.shape[0])

    def test_n_none_returns_every_person_that_passed_the_filters(self):
        # a narrow age band keeps this to a small number of Person-objects while still
        # exercising the "no sampling at all" path
        pf = PersonFilterFactory.get_person_filter(["adult"])
        pf.add_filter("df", "age40", lambda x: x[DynamicRiskFactorsType.AGE.value] == 40)
        people = PopulationFactory.get_nhanes_people(
            n=None, year=1999, personFilters=pf, nhanesWeights=False,
        )
        nhanesDf = PopulationFactory.get_nhanesDf()
        nhanesDf = nhanesDf.loc[nhanesDf.year == 1999]
        nhanesDf = PopulationFactory.apply_person_filters_on_df(pf, nhanesDf)
        self.assertGreater(nhanesDf.shape[0], 0)
        self.assertEqual(nhanesDf.shape[0], people.shape[0])

    def test_custom_weights_without_n_is_refused(self):
        with self.assertRaises(RuntimeError):
            PopulationFactory.get_nhanes_people(
                n=None, year=1999, personFilters=_adults_filter(), customWeights=pd.Series([1.0]),
            )


class TestDistributionsTypeIsChecked(unittest.TestCase):
    """distributions is combined with '&' in the argument checks, so a non-bool has to be refused
       up front rather than raising an opaque TypeError out of the operator or slipping through."""

    def test_non_bool_distributions_is_refused(self):
        for bad in ("yes", 1, None, dict()):
            with self.assertRaises(RuntimeError):
                PopulationFactory.check_nhanes_people_arguments(n=10, year=1999, distributions=bad)

    def test_numpy_bool_distributions_is_accepted(self):
        PopulationFactory.check_nhanes_people_arguments(n=10, year=1999, distributions=np.True_)


class TestTopUpReachesTargetN(unittest.TestCase):
    """The top-up used to run only on the nhanesWeights branch, so the other two sampling
       modes returned fewer than n people whenever a person-level filter dropped some."""

    def test_unweighted_sampling_reaches_target_n(self):
        pf = _adults_filter_with_person_filter("ageAtLeast60", lambda x: x._age[0] >= 60)
        people = PopulationFactory.get_nhanes_people(
            n=20, year=1999, personFilters=pf, nhanesWeights=False,
        )
        self.assertEqual(20, people.shape[0])
        for person in people:
            self.assertGreaterEqual(person._age[0], 60)

    def test_custom_weights_reaches_target_n(self):
        nhanesDf = PopulationFactory.get_nhanesDf()
        customWeights = (nhanesDf.gender == NHANESGender.MALE.value).astype(float)
        pf = _adults_filter_with_person_filter("ageAtLeast60", lambda x: x._age[0] >= 60)
        people = PopulationFactory.get_nhanes_people(
            n=20, year=1999, personFilters=pf, customWeights=customWeights,
        )
        self.assertEqual(20, people.shape[0])


class TestTopUpUsesSamplingWeights(unittest.TestCase):
    """The top-up used to sample without weights, so the people it added came from a
       different distribution than the people of the initial, weighted draw."""

    def test_zero_weight_rows_are_never_drawn_by_the_top_up(self):
        # females get a weight of exactly 0, so a weight-respecting top-up can never
        # produce a female, no matter how many rounds it needs
        nhanesDf = PopulationFactory.get_nhanesDf()
        customWeights = (nhanesDf.gender == NHANESGender.MALE.value).astype(float)
        # this person filter rejects the large majority of NHANES, so most of the returned
        # people come from the top-up rather than from the initial draw
        pf = _adults_filter_with_person_filter("ageAtLeast60", lambda x: x._age[0] >= 60)
        people = PopulationFactory.get_nhanes_people(
            n=30, year=1999, personFilters=pf, customWeights=customWeights,
        )
        self.assertEqual(30, people.shape[0])
        for person in people:
            self.assertEqual(NHANESGender.MALE, person._gender)


class TestTopUpTerminates(unittest.TestCase):
    """An impossible person filter used to make bring_people_to_target_n loop forever."""

    def setUp(self):
        nhanesDf = PopulationFactory.get_nhanesDf()
        nhanesDf = nhanesDf.loc[nhanesDf.year == 1999]
        self.df = PopulationFactory.apply_person_filters_on_df(_adults_filter(), nhanesDf).head(5)
        self.imr = InitializationModelRepository()
        self.emptyPeople = pd.Series([], dtype=object)

    def test_impossible_person_filter_raises_instead_of_hanging(self):
        pf = PersonFilterFactory.get_person_filter([])
        pf.add_filter("person", "rejectEveryone", lambda x: False)
        # maxDraws keeps this test to a handful of Person-objects; the default budget would
        # be max(100*n, 500)
        with self.assertRaises(RuntimeError):
            PopulationFactory.bring_people_to_target_n(
                5, self.emptyPeople, self.df, pf, popType=PopulationType.NHANES.value,
                initializationModelRepository=self.imr, maxDraws=20,
            )

    def test_empty_dataframe_raises(self):
        with self.assertRaises(RuntimeError):
            PopulationFactory.bring_people_to_target_n(
                5, self.emptyPeople, self.df.head(0), _adults_filter(),
                popType=PopulationType.NHANES.value, initializationModelRepository=self.imr,
            )


if __name__ == "__main__":
    unittest.main()
